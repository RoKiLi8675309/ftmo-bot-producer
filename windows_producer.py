import os
import sys
import time
import json
import logging
import threading
import psutil
import math
import queue
import gc
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# --- PATH SAFETY FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Third-Party Imports
import pytz
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import redis

# MT5 Import (Windows Only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    print("CRITICAL: MetaTrader5 package not found. pip install MetaTrader5")
    sys.exit(1)

# --- PROJECT IMPORTS ---
try:
    from shared import (
        CONFIG,
        setup_logging, LogSymbols,
        get_redis_connection,
        FTMORiskMonitor,
        SessionGuard,
        NewsEventMonitor,
        FTMOComplianceGuard,
        PrecisionGuard,
        ClusterContextBuilder,
        TimeFeatureTransformer
    )
except ImportError as e:
    print(f"CRITICAL: Failed to import 'shared' module. Ensure you are running from the project root.\nError: {e}")
    sys.exit(1)

# Initialize Logging
setup_logging("WindowsProducer")
log = logging.getLogger("Producer")

# --- CONFIGURATION CONSTANTS ---
SYMBOLS = CONFIG['trading'].get('symbols', [])
AUX_SYMBOLS = CONFIG['trading'].get('auxiliary_symbols', [])
ALL_MONITORED_SYMBOLS = list(set(SYMBOLS + AUX_SYMBOLS))

STREAM_KEY = CONFIG['redis']['price_data_stream']
TRADE_REQUEST_STREAM = CONFIG['redis']['trade_request_stream']
CLOSED_TRADE_STREAM = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
MAGIC_NUMBER = CONFIG['trading']['magic_number']
KILL_SWITCH_FILE = "kill_switch.lock"

# Safety: Max allowed latency for a signal before it is deemed STALE
MAX_TRADE_LATENCY_SECONDS = 5.0

# Dynamic Timeframe Mapping
TARGET_TF_STR = CONFIG['trading'].get('timeframe', 'M5').upper()
try:
    TIMEFRAME_MT5 = getattr(mt5, f"TIMEFRAME_{TARGET_TF_STR}")
    log.info(f"TIMEFRAME CONFIG: Set to {TARGET_TF_STR} (MT5 Constant: {TIMEFRAME_MT5})")
except AttributeError:
    log.warning(f"Invalid timeframe '{TARGET_TF_STR}' in config. Defaulting to M5.")
    TIMEFRAME_MT5 = mt5.TIMEFRAME_M5
    TARGET_TF_STR = "M5"

# --- ADAPTIVE TTL MANAGER ---
class AdaptiveTTLManager:
    def __init__(self, base_ttl=5.0, max_ttl=10.0, alpha=0.1):
        self.base_ttl = base_ttl
        self.max_ttl = max_ttl
        self.alpha = alpha
        self.avg_latency = 0.5
        self.latency_var = 0.0
        self.current_ttl = base_ttl
        self.lock = threading.Lock()
        
    def update(self, latency: float):
        with self.lock:
            measured_lat = max(0.0, latency)
            delta = measured_lat - self.avg_latency
            self.avg_latency += self.alpha * delta
            self.latency_var = (1 - self.alpha) * self.latency_var + self.alpha * (delta ** 2)
            jitter = math.sqrt(self.latency_var)
            
            if self.avg_latency < 2.0 and jitter < 1.0:
                self.current_ttl = self.base_ttl
            else:
                self.current_ttl = min(self.max_ttl, self.avg_latency + (3 * jitter))

    def get_ttl(self) -> float:
        return self.current_ttl

class MT5ExecutionEngine:
    def __init__(self, redis_client, lock: threading.RLock, risk_monitor: FTMORiskMonitor):
        self.lock = lock
        self.default_deviation = CONFIG['trading'].get('slippage', 5)
        self.magic_number = CONFIG['trading']['magic_number']
        self.r = redis_client
        self.risk_monitor = risk_monitor
        self.broker_time_offset = 0.0
        
        # Local Cache for In-Flight Orders (Anti-Race Condition)
        self.inflight_orders = set() 
        self.inflight_lock = threading.Lock()

        # Symbol Mapping (Raw -> Broker Specific)
        self.symbol_map: Dict[str, str] = {}

        if not self._check_connection():
             log.critical("CRITICAL: MT5 Not Connected. Aborting.")
             raise RuntimeError("MT5 Connection Failed")

        # 1. Build Suffix Map first
        self._build_symbol_mapping()

        # 2. Sync Time
        if not self._calculate_broker_offset_robust():
            log.critical("CRITICAL: Timezone Sync Failed. Aborting startup.")
            raise RuntimeError("Timezone Sync Failed")

    def _check_connection(self) -> bool:
        """Checks connection with a timeout guard."""
        # Note: We avoid holding the main lock here to prevent deadlocks if MT5 hangs
        return mt5.terminal_info().connected

    def _build_symbol_mapping(self):
        log.info("🔍 Scanning for Broker Symbol Suffixes & Forcing Market Watch...")
        with self.lock:
            all_symbols = mt5.symbols_get()
            if not all_symbols:
                log.warning("⚠️ No symbols found in MT5 Market Watch! Check Terminal.")
                return

            broker_symbols = {s.name: s for s in all_symbols}
            
            for raw_sym in ALL_MONITORED_SYMBOLS:
                matched_sym = None
                
                # Direct Match
                if raw_sym in broker_symbols:
                    matched_sym = raw_sym
                else:
                    # Suffix Match
                    for b_sym in broker_symbols:
                        if b_sym.startswith(raw_sym):
                            suffix = b_sym[len(raw_sym):]
                            if len(suffix) <= 4:
                                matched_sym = b_sym
                                break
                
                if matched_sym:
                    self.symbol_map[raw_sym] = matched_sym
                    # CRITICAL: Force enable in Market Watch
                    if not mt5.symbol_select(matched_sym, True):
                        log.error(f"❌ Failed to enable {matched_sym} in Market Watch!")
                    else:
                        log.info(f"🔗 Mapped {raw_sym} -> {matched_sym} (Selected)")
                else:
                    log.error(f"❌ Could not find broker symbol for {raw_sym}!")
                    self.symbol_map[raw_sym] = raw_sym 

    def _calculate_broker_offset_robust(self, max_retries: int = 10) -> bool:
        log.info("Calculating Broker-Local Time Offset...")
        candidates = [self.symbol_map.get(s, s) for s in SYMBOLS]
        if not candidates:
            candidates = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD"]
        unique_candidates = list(dict.fromkeys(candidates))
        
        for attempt in range(max_retries):
            with self.lock:
                for sym in unique_candidates:
                    if not mt5.symbol_select(sym, True):
                        continue
                    tick = mt5.symbol_info_tick(sym)
                    if tick:
                        server_ts = tick.time
                        local_ts = datetime.now(timezone.utc).timestamp()
                        self.broker_time_offset = server_ts - local_ts
                        try: self.r.set("producer:broker_time_offset", self.broker_time_offset)
                        except: pass
                        log.info(f"TIMEZONE AUDIT: Broker Offset: {self.broker_time_offset:.2f}s (via {sym})")
                        return True
            log.warning(f"Timezone Sync Attempt {attempt+1}/{max_retries} failed. Retrying...")
            time.sleep(2)
        return False

    def _get_symbol_info(self, symbol: str):
        broker_sym = self.symbol_map.get(symbol, symbol)
        with self.lock:
            info = mt5.symbol_info(broker_sym)
            if info and not info.visible:
                mt5.symbol_select(broker_sym, True)
                info = mt5.symbol_info(broker_sym)
            return info

    def _check_idempotency(self, symbol: str, unique_id: str) -> Optional[Dict[str, Any]]:
        if not unique_id: return None
        
        broker_sym = self.symbol_map.get(symbol, symbol)
        search_token = unique_id[:8]
        
        with self.lock:
            # 1. Check Positions (Live Trades)
            positions = mt5.positions_get(symbol=broker_sym)
            if positions:
                for pos in positions:
                    if search_token in pos.comment:
                        log.warning(f"🛑 IDEMPOTENCY GUARD: Signal {unique_id} already executing as Position {pos.ticket}.")
                        return {"retcode": mt5.TRADE_RETCODE_DONE, "order": pos.ticket, "price": pos.price_open}
            
            # 2. Check Orders (Pending Limits/Stops)
            orders = mt5.orders_get(symbol=broker_sym)
            if orders:
                for order in orders:
                    if search_token in order.comment:
                        log.warning(f"🛑 IDEMPOTENCY GUARD: Signal {unique_id} found as Pending Order {order.ticket}.")
                        return {"retcode": mt5.TRADE_RETCODE_PLACED, "order": order.ticket}
                        
        return None

    def _check_hard_trade_limit(self, symbol: str) -> bool:
        """
        HARD SOURCE OF TRUTH CHECK + IN-FLIGHT CACHE.
        Includes a local cache check to prevent race conditions where
        multiple threads pass this check before the first trade is recorded by MT5.
        """
        max_trades = CONFIG['risk_management'].get('max_open_trades', 1)
        
        try:
            # 1. Check In-Flight Cache (Trades sent but not yet in MT5)
            with self.inflight_lock:
                inflight_count = len(self.inflight_orders)
                if inflight_count >= max_trades:
                    log.critical(f"🛑 HARD LIMIT GATE (IN-FLIGHT): {inflight_count} orders currently processing. Trade Blocked.")
                    return False

            with self.lock:
                # 2. Get ALL positions for this bot (Single Source of Truth)
                positions = mt5.positions_get()
                
                # Check for MT5 Error - FAIL CLOSED
                if positions is None:
                    err = mt5.last_error()
                    log.error(f"❌ MT5 ERROR: positions_get() returned None. Error: {err}. Blocking trade for safety.")
                    return False
                
                # Filter for this bot's magic number
                bot_positions = [p for p in positions if p.magic == self.magic_number]
                count = len(bot_positions)
                
                # Global Limit Check
                total_utilization = count + inflight_count
                
                if total_utilization >= max_trades:
                    log.critical(f"🛑 HARD LIMIT GATE: Live ({count}) + In-Flight ({inflight_count}) >= Limit ({max_trades}). Trade Blocked.")
                    return False
                
                return True
                
        except Exception as e:
            log.error(f"Hard Limit Check Failed: {e}", exc_info=True)
            return False # Fail safe: Block if we can't verify

    def execute_trade(self, request: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        log.info(f"🏗️ BUILDING ORDER: {request.get('symbol')} {request.get('type')}")
        
        raw_symbol = request.get("symbol")
        if not raw_symbol: 
            log.error("EXECUTION FAIL: No symbol in request.")
            return None
            
        # 1. Resolve Symbol
        broker_sym = self.symbol_map.get(raw_symbol, raw_symbol)
        request["symbol"] = broker_sym
        
        # 2. Get Info
        symbol_info = self._get_symbol_info(raw_symbol)
        if not symbol_info:
            log.error(f"EXECUTION FAIL: Could not get symbol_info for {broker_sym}. Is it in Market Watch?")
            return None
        
        raw_comment = str(request.get("comment", ""))
        signal_uuid = request.get('uuid', '')
        if not signal_uuid and "Auto_" in raw_comment:
             parts = raw_comment.split('_')
             if len(parts) > 1: signal_uuid = parts[1]

        # --- PRE-FLIGHT IDEMPOTENCY CHECK ---
        if signal_uuid:
            existing = self._check_idempotency(raw_symbol, signal_uuid)
            if existing:
                log.info(f"✅ Trade {signal_uuid} already exists. Skipping duplicate.")
                return existing

        # --- FORCE MARKET EXECUTION (AGGRESSOR PROTOCOL) ---
        # No more limit logic. Always execute at market.
        
        try:
            with self.lock:
                term_info = mt5.terminal_info()
                if not term_info.trade_allowed:
                    log.critical("🚨 ALGO TRADING DISABLED IN TERMINAL! Please enable 'Algo Trading' button in MT5.")
                
                tick = mt5.symbol_info_tick(broker_sym)
                if not tick: 
                    log.error(f"EXECUTION FAIL: No tick data for {broker_sym}.")
                    return None
            
            # --- MARKET ORDER CONSTRUCTION ---
            if request["action"] != mt5.TRADE_ACTION_SLTP: # Don't override SL/TP modifications
                request["action"] = mt5.TRADE_ACTION_DEAL
                request["type_time"] = mt5.ORDER_TIME_GTC
                
                # ALWAYS use current tick price for Market Execution
                if request["type"] == mt5.ORDER_TYPE_BUY:
                    request["price"] = tick.ask
                elif request["type"] == mt5.ORDER_TYPE_SELL:
                    request["price"] = tick.bid
                
                request["price"] = PrecisionGuard.normalize_price(request["price"], broker_sym, symbol_info)

            # --- HARD LIMIT CHECK (SOURCE OF TRUTH) ---
            # Perform the check ONLY for Entries (Deals or Pending Orders)
            if request["action"] in [mt5.TRADE_ACTION_DEAL, mt5.TRADE_ACTION_PENDING]:
                if not self._check_hard_trade_limit(broker_sym):
                    return None # BLOCK TRADING

            raw_vol = float(request.get("volume", 0.01))
            
            vol_step = symbol_info.volume_step
            if vol_step > 0:
                steps = round(raw_vol / vol_step)
                request["volume"] = steps * vol_step
            request["volume"] = max(symbol_info.volume_min, min(request["volume"], symbol_info.volume_max))
            request["volume"] = round(request["volume"], 2)

            raw_sl = float(request.get("sl", 0.0))
            if raw_sl > 0:
                request["sl"] = PrecisionGuard.normalize_price(raw_sl, broker_sym, symbol_info)
            
            raw_tp = float(request.get("tp", 0.0))
            if raw_tp > 0:
                request["tp"] = PrecisionGuard.normalize_price(raw_tp, broker_sym, symbol_info)
            
            # Since we are MARKET ONLY, filling type is usually FOK or IOC
            if request["action"] == mt5.TRADE_ACTION_PENDING:
                 # Should not happen in Market Only mode, but kept for safety if Pending passed manually
                request["type_filling"] = mt5.ORDER_FILLING_RETURN
            else:
                filling = symbol_info.filling_mode
                if filling & 1: request["type_filling"] = mt5.ORDER_FILLING_FOK
                elif filling & 2: request["type_filling"] = mt5.ORDER_FILLING_IOC
                else: request["type_filling"] = mt5.ORDER_FILLING_RETURN
            
            request['action'] = int(request['action'])
            request['type'] = int(request['type'])
            request['volume'] = float(request['volume'])
            request['price'] = float(request['price'])
            if 'sl' in request: request['sl'] = float(request['sl'])
            if 'tp' in request: request['tp'] = float(request['tp'])
            if 'magic' in request: request['magic'] = int(request['magic'])
            if 'type_time' in request: request['type_time'] = int(request['type_time'])
            if 'type_filling' in request: request['type_filling'] = int(request['type_filling'])

            safe_comment = raw_comment.replace("|", " ").replace(":", " ").replace("%", "").replace("_", "")
            if signal_uuid:
                short_uuid = signal_uuid[:8]
                final_comment = f"{short_uuid} {safe_comment}"
            else:
                final_comment = safe_comment
            request["comment"] = final_comment[:23].strip()

        except (ValueError, TypeError) as e:
            log.error(f"Sanitization/Casting error for {broker_sym}: {e}")
            return None

        # --- EXECUTION LOOP ---
        # REGISTER IN-FLIGHT
        if signal_uuid:
            with self.inflight_lock:
                self.inflight_orders.add(signal_uuid)

        try:
            log.info(f"🚀 SENDING ORDER TO MT5: {json.dumps(request, default=str)}")
            for attempt in range(max_retries):
                if attempt > 0 and signal_uuid:
                    existing = self._check_idempotency(raw_symbol, signal_uuid)
                    if existing:
                        log.info(f"✅ Trade {signal_uuid} already exists. Skipping duplicate.")
                        return existing

                # UNLOCKED EXECUTION:
                # We assume request is fully prepared. MT5 SDK is generally thread-safe for order_send
                result = mt5.order_send(request)
                
                if result is None:
                    err = mt5.last_error()
                    log.warning(f"MT5 Order Send returned None. LAST ERROR: {err}. Checking idempotency...")
                    
                    log.warning("🔄 FORCING MT5 RECONNECT...")
                    with self.lock:
                        mt5.shutdown()
                        time.sleep(0.5)
                        mt5.initialize()
                    
                    if signal_uuid:
                        ghost = self._check_idempotency(raw_symbol, signal_uuid)
                        if ghost: return ghost
                    
                    time.sleep(0.5)
                    continue
                    
                log.info(f"MT5 RESPONSE: Retcode={result.retcode} Comment='{result.comment}' Ticket={result.order}")

                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    log.info(f"✅ EXECUTION SUCCESS: {broker_sym} Ticket: {result.order}")
                    return result._asdict()
                elif result.retcode == mt5.TRADE_RETCODE_PLACED:
                    log.info(f"✅ LIMIT ORDER PLACED: {broker_sym} Ticket: {result.order} @ {request['price']}")
                    return result._asdict()
                
                elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_CONNECTION, mt5.TRADE_RETCODE_PRICE_OFF, mt5.TRADE_RETCODE_INVALID_PRICE, 10016]:
                    log.warning(f"Recoverable Error ({result.retcode}). Retrying...")
                    time.sleep(0.5 * (2 ** attempt))
                    
                    with self.lock:
                        new_tick = mt5.symbol_info_tick(broker_sym)
                        if new_tick:
                            # REFRESH PRICE FOR RETRY
                            if request["action"] == mt5.TRADE_ACTION_DEAL:
                                if request["type"] == mt5.ORDER_TYPE_BUY:
                                    request["price"] = new_tick.ask
                                elif request["type"] == mt5.ORDER_TYPE_SELL:
                                    request["price"] = new_tick.bid
                                
                                request["price"] = PrecisionGuard.normalize_price(request["price"], broker_sym, symbol_info)

                    continue
                else:
                    log.error(f"❌ EXECUTION FAILURE: {broker_sym} Retcode: {result.retcode} ({result.comment})")
                    try: self.r.publish("order_failed_channel", json.dumps({"symbol": raw_symbol, "reason": result.comment}))
                    except: pass
                    break
            return None
        finally:
            # CLEANUP IN-FLIGHT
            if signal_uuid:
                with self.inflight_lock:
                    self.inflight_orders.discard(signal_uuid)

    def close_position(self, position_id: int, symbol: str, volume: float, pos_type: int) -> Optional[Any]:
        with self.lock:
            tick = mt5.symbol_info_tick(symbol)
            if not tick: return None
            price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
            trade_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
                "type": trade_type, "position": position_id, "price": price,
                "deviation": self.default_deviation, "magic": self.magic_number,
                "comment": "Algo Close", "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC
            }
            return mt5.order_send(request)

class HybridProducer:
    def __init__(self):
        self.running = True
        self.stop_event = threading.Event()
        
        log.info("🔌 Connecting to Redis...")
        try:
            self.r = get_redis_connection(host=CONFIG['redis']['host'], port=CONFIG['redis']['port'], db=0, decode_responses=True)
            self.r.set("producer:probe", "alive")
            val = self.r.get("producer:probe")
            if val == "alive":
                log.info(f"✅ REDIS CONNECTED SUCCESSFULLY: {CONFIG['redis']['host']}:{CONFIG['redis']['port']}")
                
                stream_key = CONFIG['redis']['trade_request_stream']
                log.info(f"🔌 REDIS CONFIG: Host={CONFIG['redis']['host']} Port={CONFIG['redis']['port']} Stream={stream_key}")
                
                if self.r.exists(stream_key):
                    slen = self.r.xlen(stream_key)
                    log.info(f"✅ Stream '{stream_key}' found. Length: {slen}")
                else:
                    log.warning(f"⚠️ Stream '{stream_key}' does NOT exist yet (Normal if no trades sent).")
                    
            else:
                raise RuntimeError("Redis Probe Failed")
        except Exception as e:
            log.critical(f"❌ REDIS CONNECTION FAILURE: {e}")
            sys.exit(1)

        self.mt5_lock = threading.RLock()
        self.execution_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="BgWorker")
        
        self.db_dsn = CONFIG['postgres']['dsn']
        self._connect_db_with_retry()
        initial_bal = CONFIG.get('env', {}).get('initial_balance', 100000.0)
        self.ftmo_monitor = FTMORiskMonitor(initial_balance=initial_bal, max_daily_loss_pct=CONFIG['risk_management']['max_daily_loss_pct'], redis_client=self.r)
        self._optimize_process()
        self.connect_mt5()
        self.exec_engine = MT5ExecutionEngine(self.r, self.mt5_lock, self.ftmo_monitor)
        self.ttl_manager = AdaptiveTTLManager()
        self.session_guard = SessionGuard()
        self.news_monitor = NewsEventMonitor()
        self.compliance_guard = FTMOComplianceGuard([])
        self.time_engine = TimeFeatureTransformer()
        self.cluster_engine = ClusterContextBuilder(SYMBOLS)
        self.d1_cache = {p: {} for p in SYMBOLS}
        self.h4_cache = {p: {} for p in SYMBOLS}
        self.last_context_update = 0
        self.notified_tickets = set()
        
        self.last_tick_state = defaultdict(lambda: {'time_msc': 0, 'volume_real': 0.0})
        self.last_prices = {s: 0.0 for s in SYMBOLS}
        
        # Initialize last_deal_scan_time to 24h ago in SERVER TIME to ensure we catch up
        self.last_deal_scan_server_ts = 0.0 
        
        self.run_precise_backfill()
        self._reconstruct_risk_state_from_history()
        self.monitored_symbols = self._ensure_conversion_pairs()
        self.monitored_price_keys = {s: f"price:{s}" for s in self.monitored_symbols}

    def _connect_db_with_retry(self, max_retries=10):
        log.info("Connecting to Database...")
        for attempt in range(max_retries):
            try:
                self.conn = psycopg2.connect(self.db_dsn)
                self.conn.autocommit = True
                log.info("Database Connected Successfully.")
                return
            except Exception as e:
                log.warning(f"DB Connection Attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(5)
        log.critical("Could not connect to Database.")
        sys.exit(1)

    @contextmanager
    def db_cursor(self):
        try:
            if self.conn.closed: self._connect_db_with_retry(3)
            yield self.conn.cursor()
        except Exception as e:
            log.error(f"DB Cursor Error: {e}")
            self._connect_db_with_retry(3)
            yield self.conn.cursor()

    def _optimize_process(self):
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            p_cores = list(range(16))
            try:
                proc.cpu_affinity(p_cores)
                log.info(f"CPU Affinity set to P-Cores: {p_cores}")
            except Exception as e:
                log.warning(f"Could not set CPU affinity: {e}")
        except Exception as e:
            log.warning(f"Process optimization failed: {e}")

    def connect_mt5(self):
        with self.mt5_lock:
            mt5_conf = CONFIG.get('mt5', {})
            path = mt5_conf.get('path')
            login = mt5_conf.get('login')
            password = mt5_conf.get('password')
            server = mt5_conf.get('server')
            init_params = {}
            if path and os.path.exists(path): init_params['path'] = path
            if not mt5.initialize(**init_params):
                log.critical(f"MT5 Initialize Failed: {mt5.last_error()}")
                sys.exit(1)
            if login and password and server:
                try:
                    if not mt5.login(login=int(login), password=password, server=server):
                        log.critical(f"MT5 Login Failed: {mt5.last_error()}")
                        sys.exit(1)
                    log.info(f"MT5 Logged in as {login} on {server}")
                except Exception as e:
                    log.error(f"MT5 Login Exception: {e}")
                    sys.exit(1)
            else:
                log.warning("MT5 Credentials missing in Config. Running in initialized mode only.")
            for sym in ALL_MONITORED_SYMBOLS:
                if not mt5.symbol_select(sym, True): 
                    log.warning(f"Failed to select raw symbol {sym}")
                else: 
                    log.info(f"Subscribed to {sym}")

    def _reconstruct_risk_state_from_history(self):
        log.info("Reconstructing Risk State (FTMO Compliance - Server Time Authority)...")
        retry_count = 0
        max_retries = 5
        
        # Load Risk Timezone from Config (Default Europe/Prague for FTMO)
        risk_tz_str = CONFIG.get('risk_management', {}).get('risk_timezone', 'Europe/Prague')
        try:
            risk_tz = pytz.timezone(risk_tz_str)
        except:
            risk_tz = pytz.timezone('Europe/Prague')
            
        while retry_count < max_retries:
            try:
                with self.mt5_lock:
                    if not mt5.terminal_info().connected:
                        log.warning(f"MT5 Disconnected. Retry {retry_count+1}/{max_retries}...")
                        time.sleep(2)
                        retry_count += 1
                        continue
                    info = mt5.account_info()
                    if not info:
                        log.warning(f"Failed to get Account Info. Retry {retry_count+1}...")
                        retry_count += 1
                        time.sleep(2)
                        continue
                    current_balance = info.balance
                    if abs(self.ftmo_monitor.initial_balance - current_balance) > (current_balance * 0.01):
                        log.warning(f"⚠️ Auto-Detecting Account Size: Config ({self.ftmo_monitor.initial_balance}) != Broker ({current_balance}). Updating Risk Limits.")
                        self.ftmo_monitor.initial_balance = current_balance
                        self.ftmo_monitor.max_daily_loss = current_balance * CONFIG['risk_management']['max_daily_loss_pct']
                    
                    # AUDIT FIX: Get Accurate Midnight Anchor using Broker Time
                    # We need the current Server Time.
                    # We query a robust symbol tick to get the latest server timestamp.
                    server_ts = time.time() # Fallback
                    for s in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]:
                         tick = mt5.symbol_info_tick(s)
                         if tick:
                             server_ts = tick.time
                             break
                    
                    # FTMO (and most brokers) resets daily stats at 00:00 CE(S)T.
                    # We must align our "Start of Day" with the Risk Timezone.
                    # Convert Server TS (Epoch) to Risk TZ
                    dt_utc = datetime.fromtimestamp(server_ts, timezone.utc)
                    dt_risk = dt_utc.astimezone(risk_tz)
                    
                    # Find Midnight in Risk TZ
                    dt_midnight_risk = dt_risk.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    # Convert back to UTC Epoch for MT5 query
                    dt_midnight_utc = dt_midnight_risk.astimezone(timezone.utc)
                    midnight_ts = dt_midnight_utc.timestamp()
                    
                    # History Deals: FROM midnight TO now
                    # Add buffer to 'now' to catch everything
                    deals = mt5.history_deals_get(float(midnight_ts), float(server_ts + 3600))
                    
                    realized_pnl_today = 0.0
                    if deals:
                        for d in deals:
                            if d.magic == MAGIC_NUMBER: realized_pnl_today += (d.profit + d.swap + d.commission)
                    else:
                        log.info("No deals found for today (Start of Day).")
                    
                    # Reconstruct Start Equity
                    # Start = Current - PnL
                    # Example: Current=10100, PnL=+100 -> Start=10000
                    calculated_start = current_balance - realized_pnl_today
                    self.ftmo_monitor.starting_equity_of_day = calculated_start
                    self.r.set(CONFIG['redis']['risk_keys']['daily_starting_equity'], calculated_start)
                    self.r.set("bot:account_size", self.ftmo_monitor.initial_balance) 
                    
                    # Persist the Anchor Date to allow Engine to detect rollover
                    self.r.set("risk:last_reset_date", str(midnight_ts)) 
                    
                    loss_limit = calculated_start * CONFIG['risk_management']['max_daily_loss_pct']
                    hard_deck = calculated_start - loss_limit
                    self.r.set("risk:hard_deck_level", hard_deck)
                    log.info(f"{LogSymbols.SUCCESS} RISK STATE VERIFIED: Start Equity: {calculated_start:.2f} | Hard Deck: {hard_deck:.2f} | PnL Today: {realized_pnl_today:.2f} (Timezone: {risk_tz_str})")
                    return
            except Exception as e:
                log.error(f"Risk Reconstruction Exception: {e}")
                retry_count += 1
                time.sleep(2)
        log.critical("CRITICAL: FAILED TO VERIFY RISK STATE AFTER 5 ATTEMPTS. ABORTING STARTUP TO PREVENT COMPLIANCE BREACH.")
        sys.exit(1)

    def run_precise_backfill(self):
        log.warning("=== STARTING PRECISE DATA BACKFILL ===")
        self.ensure_ohlcv_table()
        start_str = CONFIG['data'].get('download_start_date', '2020-01-01')
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            utc_from = pytz.utc.localize(start_dt)
            utc_now = datetime.now(pytz.utc)
            chunk_days = 30
            current_start = utc_from
            while current_start < utc_now:
                current_end = current_start + timedelta(days=chunk_days)
                if current_end > utc_now: current_end = utc_now
                with self.db_cursor() as cur:
                    for sym in SYMBOLS:
                        broker_sym = self.exec_engine.symbol_map.get(sym, sym)
                        with self.mt5_lock:
                            rates = mt5.copy_rates_range(broker_sym, TIMEFRAME_MT5, current_start, current_end)
                            if rates is None or len(rates) == 0: continue
                            data_tuples = []
                            for r in rates:
                                try:
                                    ts_val = r['time']
                                    if ts_val <= 0: continue
                                    row_time = pd.to_datetime(ts_val, unit='s', utc=True).isoformat()
                                    data_tuples.append((
                                        row_time, sym, TARGET_TF_STR,
                                        float(r['open']), float(r['high']), float(r['low']), float(r['close']),
                                        int(r['tick_volume'])
                                    ))
                                except: continue
                            if data_tuples:
                                execute_values(cur, """
                                    INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                                    VALUES %s
                                    ON CONFLICT (time, symbol, timeframe) DO NOTHING
                                """, data_tuples)
                current_start = current_end
                gc.collect()
            log.info(f"{LogSymbols.SUCCESS} Data Backfill Complete.")
        except Exception as e:
            log.error(f"Backfill Failed: {e}")

    def ensure_ohlcv_table(self):
        sql = """
        CREATE TABLE IF NOT EXISTS ohlcv (
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            PRIMARY KEY (time, symbol, timeframe)
        );
        """
        with self.db_cursor() as cur:
            cur.execute(sql)
            self.conn.commit()

    def _ensure_conversion_pairs(self) -> Set[str]:
        monitored = set(ALL_MONITORED_SYMBOLS)
        with self.mt5_lock:
            account_info = mt5.account_info()
            if not account_info: return monitored
            acc_ccy = account_info.currency
            for sym in SYMBOLS:
                broker_sym = self.exec_engine.symbol_map.get(sym, sym)
                info = mt5.symbol_info(broker_sym)
                if not info: continue
                profit_ccy = info.currency_profit
                if profit_ccy != acc_ccy:
                    candidates = [f"{profit_ccy}{acc_ccy}", f"{acc_ccy}{profit_ccy}"]
                    for c in candidates:
                        if mt5.symbol_select(c, True):
                            monitored.add(c)
                            break
        return monitored

    def _estimate_flow_volumes(self, symbol: str, current_vol: float, current_price: float) -> Tuple[float, float]:
        try:
            last_price = self.last_prices.get(symbol, current_price)
            if last_price == 0: return current_vol / 2.0, current_vol / 2.0
            
            if current_price > last_price: return float(current_vol), 0.0
            elif current_price < last_price: return 0.0, float(current_vol)
            else:
                half_vol = float(current_vol) / 2.0
                return half_vol, half_vol
        except Exception:
            return 0.0, 0.0

    def _tick_stream_loop(self):
        log.info(f"{LogSymbols.ONLINE} Starting Deduplicated Tick Stream...")
        interval = CONFIG['producer']['tick_interval_seconds']
        redis_failures = 0
        print("\n\n=== PRODUCER STREAMING STARTED ===\n\n")
        
        # WATCHDOG FOR ZOMBIE TERMINAL
        last_successful_tick_time = time.time()
        
        while self.running and not self.stop_event.is_set():
            if os.path.exists(KILL_SWITCH_FILE):
                log.critical("💀 KILL SWITCH DETECTED. SHUTTING DOWN IMMEDIATELY.")
                with self.mt5_lock: mt5.shutdown()
                sys.exit(0)
            start = time.time()
            try:
                # Watchdog Check
                if time.time() - last_successful_tick_time > 60:
                    log.warning("⚠️ No ticks for 60s! Attempting MT5 Reconnect...")
                    with self.mt5_lock:
                        mt5.shutdown()
                        time.sleep(1)
                        mt5.initialize()
                        last_successful_tick_time = time.time() # Reset watchdog

                if time.time() - self.last_context_update > 60:
                    self._update_d1_context()
                    self._update_h4_context()
                    self.last_context_update = time.time()
                
                pipe = self.r.pipeline()
                updates_count = 0
                for sym in self.monitored_symbols:
                    broker_sym = self.exec_engine.symbol_map.get(sym, sym)
                    with self.mt5_lock:
                        tick = mt5.symbol_info_tick(broker_sym)
                        if not tick: continue
                        last_state = self.last_tick_state[sym]
                        raw_vol = float(tick.volume_real if tick.volume_real > 0 else tick.volume)
                        current_vol = raw_vol if raw_vol > 0 else 1.0
                        
                        if tick.time_msc == last_state['time_msc'] and current_vol == last_state['volume_real']:
                            continue
                        
                        self.last_tick_state[sym]['time_msc'] = tick.time_msc
                        self.last_tick_state[sym]['volume_real'] = current_vol
                        
                        sym_info = mt5.symbol_info(broker_sym)
                        digits = sym_info.digits if sym_info else 5
                        bid = round(tick.bid, digits)
                        ask = round(tick.ask, digits)
                        price_now = tick.last if tick.last > 0 else (bid + ask) / 2.0
                        
                        bid_vol, ask_vol = self._estimate_flow_volumes(sym, current_vol, price_now)
                        if bid_vol == 0 and ask_vol == 0:
                            bid_vol = current_vol / 2.0
                            ask_vol = current_vol / 2.0
                        
                        # Correctly update last price AFTER estimation
                        self.last_prices[sym] = price_now
                        
                        self.cluster_engine.update_correlations(pd.DataFrame())
                        
                        utc_ts = int(tick.time_msc) - int(self.exec_engine.broker_time_offset * 1000)
                        
                        payload = {
                            "symbol": sym, "time": utc_ts,
                            "bid": bid, "ask": ask,
                            "price": price_now, "volume": current_vol,
                            "bid_vol": float(bid_vol), "ask_vol": float(ask_vol),
                            "ctx_d1": json.dumps(self.d1_cache.get(sym, {})),
                            "ctx_h4": json.dumps(self.h4_cache.get(sym, {}))
                        }
                        
                        pipe.xadd(STREAM_KEY, payload, maxlen=10000, approximate=True)
                        updates_count += 1
                        
                        key = self.monitored_price_keys.get(sym, f"price:{sym}")
                        pipe.hset(key, mapping={"bid": payload["bid"], "ask": payload["ask"], "time": int(utc_ts/1000)})
                
                if updates_count > 0:
                    pipe.execute()
                    last_successful_tick_time = time.time()
                
                self.r.setex(CONFIG['redis']['heartbeat_key'], 10, str(time.time()))
                redis_failures = 0
            except (redis.ConnectionError, redis.TimeoutError) as e:
                redis_failures += 1
                sleep_time = min(2 ** redis_failures, 30)
                log.error(f"Redis Disconnected. Retrying in {sleep_time}s... ({e})")
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Tick Loop Error: {e}")
            elapsed = time.time() - start
            time.sleep(max(0.0, interval - elapsed))

    def _update_d1_context(self):
        for sym in SYMBOLS:
            broker_sym = self.exec_engine.symbol_map.get(sym, sym)
            with self.mt5_lock:
                rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_D1, 1, 500)
                if rates is not None and len(rates) > 200:
                    r = rates[-1]
                    closes = pd.Series([x['close'] for x in rates])
                    ema_series = closes.ewm(span=200, adjust=False).mean()
                    ema = ema_series.iloc[-1]
                    self.d1_cache[sym] = {
                        'open': float(r['open']), 'high': float(r['high']),
                        'low': float(r['low']), 'close': float(r['close']),
                        'ema200': float(ema)
                    }

    def _update_h4_context(self):
        for sym in SYMBOLS:
            broker_sym = self.exec_engine.symbol_map.get(sym, sym)
            with self.mt5_lock:
                rates = mt5.copy_rates_from_pos(broker_sym, mt5.TIMEFRAME_H4, 1, 100)
                if rates is not None and len(rates) > 14:
                    r = rates[-1]
                    df = pd.DataFrame(rates)
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    if np.isnan(current_rsi): current_rsi = 50.0
                    self.h4_cache[sym] = {
                        'open': float(r['open']), 'high': float(r['high']),
                        'low': float(r['low']), 'close': float(r['close']),
                        'rsi': float(current_rsi)
                    }

    def _trade_listener(self):
        log.info("Starting Trade Listener...")
        redis_failures = 0
        
        try:
            log.info(f"🔎 DIAGNOSTIC: Peeking into stream '{TRADE_REQUEST_STREAM}' without group...")
            peek = self.r.xrevrange(TRADE_REQUEST_STREAM, count=1)
            if peek:
                log.info(f"✅ Stream '{TRADE_REQUEST_STREAM}' is ALIVE. Last message: {peek[0][0]}")
            else:
                log.warning(f"⚠️ Stream '{TRADE_REQUEST_STREAM}' is EMPTY or does not exist yet. Waiting for dispatch...")
        except Exception as e:
            log.error(f"❌ Failed to peek stream: {e}")

        # Ensure Group Exists
        try: 
            self.r.xgroup_create(TRADE_REQUEST_STREAM, "execution_group", id='0', mkstream=True)
            log.info(f"✅ Consumer Group 'execution_group' created on '{TRADE_REQUEST_STREAM}'.")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                log.info("ℹ️ Consumer Group 'execution_group' already exists.")
            else:
                log.error(f"❌ Error creating consumer group: {e}")

        # Polling Loop
        empty_poll_counter = 0
        while self.running:
            try:
                if empty_poll_counter % 10 == 0:
                    log.info(f"💤 Polling Stream: {TRADE_REQUEST_STREAM}...")

                if int(time.time()) % 60 == 0: self.r.xtrim(TRADE_REQUEST_STREAM, maxlen=1000, approximate=True)
                
                entries = self.r.xreadgroup("execution_group", "producer_main", {TRADE_REQUEST_STREAM: '>'}, count=5, block=1000)
                
                redis_failures = 0 
                if entries:
                    empty_poll_counter = 0
                    log.info(f"📥 RAW REDIS MESSAGE: {entries}")
                    
                    for stream, messages in entries:
                        for msg_id, data in messages:
                            log.info(f"📥 PARSED TRADE REQUEST: {data}")
                            self.execution_queue.put((msg_id, data))
                else:
                    empty_poll_counter += 1

            except (redis.ConnectionError, redis.TimeoutError) as e:
                redis_failures += 1
                sleep_time = min(2 ** redis_failures, 30)
                log.error(f"Redis Listener Disconnected. Retrying in {sleep_time}s... ({e})")
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Trade Listener Generic Error: {e}", exc_info=True)
                time.sleep(1)

    def _check_mt5_revenge_status(self, symbol: str) -> bool:
        """
        Uses absolute Server Time for history lookups.
        Prevents missed trades when Local Time != Server Time.
        Acts as the 'Single Source of Truth' for Revenge Trading Cooldowns.
        """
        try:
            cooldown_mins = CONFIG.get('risk_management', {}).get('loss_cooldown_minutes', 60)
            if cooldown_mins <= 0: return False

            broker_sym = self.exec_engine.symbol_map.get(symbol, symbol)
            
            with self.mt5_lock:
                server_now_ts = time.time() + self.exec_engine.broker_time_offset
                
                # Look back 24 hours to be safe
                from_ts = server_now_ts - 86400 
                to_ts = server_now_ts + 3600 
                
                deals = mt5.history_deals_get(float(from_ts), float(to_ts))
                
                if deals:
                    relevant_deals = [
                        d for d in deals 
                        if d.symbol == broker_sym and d.magic == int(self.exec_engine.magic_number)
                    ]
                    
                    sorted_deals = sorted(relevant_deals, key=lambda x: x.time, reverse=True)
                    
                    for deal in sorted_deals:
                        time.sleep(0.001) # Yield slightly
                        time_since_close = server_now_ts - deal.time
                        
                        if time_since_close < (cooldown_mins * 60):
                            if deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                                net_pnl = deal.profit + deal.swap + deal.commission
                                expiry_in = int((cooldown_mins * 60) - time_since_close)
                                
                                if net_pnl < 0:
                                    log.warning(f"🛑 REVENGE GUARD: {symbol} BLOCKED. Loss {net_pnl:.2f} at {deal.time} (Server Time). Expiry in {expiry_in}s")
                                else:
                                    log.warning(f"🛡️ MANDATORY COOLDOWN: {symbol} BLOCKED. Win/BE {net_pnl:.2f} at {deal.time}. Expiry in {expiry_in}s")
                                
                                return True # Block ALL trades within window
                        else:
                            # Found a deal outside window, and since sorted, older ones are also outside
                            break
                            
        except Exception as e:
            log.error(f"Revenge Guard Check Failed: {e}")
            
        return False

    def _process_trade_signal_sync(self, msg_id, data):
        """
        Executed synchronously in the Main Thread loop.
        Ensures thread affinity with MT5.
        """
        try:
            symbol = data['symbol']
            action = data['action'] 
            uuid_val = data.get('uuid')
            
            price = float(data.get('price', 0.0))
            
            request_ts_str = data.get('timestamp')
            if request_ts_str:
                try:
                    request_ts = float(request_ts_str)
                    now_ts = time.time()
                    latency = abs(now_ts - request_ts)
                    self.ttl_manager.update(latency)
                    current_ttl = self.ttl_manager.get_ttl()
                    
                    if latency > MAX_TRADE_LATENCY_SECONDS:
                        log.error(f"🛑 REJECTING STALE SIGNAL: {symbol} Latency: {latency:.2f}s > Limit: {MAX_TRADE_LATENCY_SECONDS}s.")
                        self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                        return
                    
                    if latency > current_ttl:
                        log.warning(f"⚠️ HIGH LATENCY TRADE: {symbol} Latency: {latency:.2f}s > TTL: {current_ttl:.1f}s. EXECUTING ANYWAY (Within safe limit).")
                        
                except ValueError:
                    log.error(f"Invalid timestamp: {request_ts_str}")
            
            # Deduplication
            if uuid_val:
                if self.r.sismember("processed_signals", uuid_val):
                    log.warning(f"🔁 Duplicate Signal Ignored: {uuid_val}")
                    self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                    return
                self.r.sadd("processed_signals", uuid_val)
                self.r.expire("processed_signals", 3600)
            
            # --- EXECUTION ROUTING ---
            is_entry_signal = (action in ["BUY", "SELL"]) or (str(action) in ["1", "5"])
            
            if is_entry_signal:
                with self.mt5_lock:
                    info = mt5.account_info()
                    if info:
                        self.ftmo_monitor.equity = info.equity
                        if not self.ftmo_monitor.can_trade():
                            log.warning("Trade blocked by Risk Monitor.")
                            self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                            return

                # --- REVENGE GUARD CHECK ---
                if self._check_mt5_revenge_status(symbol):
                    log.warning(f"⛔ Signal Dropped: {symbol} is in cooldown (MT5 History Check).")
                    self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                    return
                
                if action in ["BUY", "SELL"]:
                    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
                    order_action = mt5.TRADE_ACTION_DEAL
                    log_type = action
                else:
                    try:
                        order_type = int(data.get('type', 0))
                        order_action = int(action)
                    except (ValueError, TypeError):
                        log.error(f"❌ MALFORMED PROTOCOL: Action='{action}' or Type='{data.get('type')}' is not integer convertible. Dropping.")
                        self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                        return

                    log_type = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"

                log.info(f"⚡ EXECUTING {log_type} {symbol} | Vol: {data['volume']} | Price: {price} | ActionCode: {order_action}")
                
                # If order type is MARKET/DEAL, force price to 0.0 to trigger Instant Execution
                exec_price = price
                if order_action == mt5.TRADE_ACTION_DEAL:
                    exec_price = 0.0

                self.exec_engine.execute_trade({
                    "action": order_action, 
                    "symbol": symbol,
                    "volume": float(data['volume']),
                    "type": order_type,
                    "price": exec_price, 
                    "sl": float(data.get("sl", 0)), 
                    "tp": float(data.get("tp", 0)),
                    "magic": MAGIC_NUMBER, 
                    "comment": data.get("comment", "Algo"),
                    "uuid": uuid_val 
                })
            
            elif action == "MODIFY":
                self.exec_engine.execute_trade({
                    "action": mt5.TRADE_ACTION_SLTP, "position": int(data['ticket']),
                    "sl": float(data.get("sl", 0)), "tp": float(data.get("tp", 0))
                })
            
            elif action == "CLOSE_ALL": 
                self._close_all_positions(symbol)
            
            self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
            
        except Exception as e: log.error(f"Trade Execution Error: {e}", exc_info=True)

    def _close_all_positions(self, symbol: str = None):
        with self.mt5_lock:
            if symbol:
                broker_sym = self.exec_engine.symbol_map.get(symbol, symbol)
                positions = mt5.positions_get(symbol=broker_sym)
            else:
                positions = mt5.positions_get()
            if not positions: return
            for p in positions:
                if p.magic != MAGIC_NUMBER: continue
                self.exec_engine.close_position(p.ticket, p.symbol, p.volume, p.type)

    def _closed_trade_monitor(self):
        log.info("Starting Closed Trade Monitor (SQN Feed - Server Time Sync)...")
        if not self.last_deal_scan_server_ts:
            self.last_deal_scan_server_ts = (time.time() + self.exec_engine.broker_time_offset) - 86400
            
        while self.running and not self.stop_event.is_set():
            try:
                server_now_ts = time.time() + self.exec_engine.broker_time_offset
                
                with self.mt5_lock: 
                    deals = mt5.history_deals_get(float(self.last_deal_scan_server_ts), float(server_now_ts + 60))
                
                self.last_deal_scan_server_ts = server_now_ts
                
                if deals:
                    pipe = self.r.pipeline()
                    published_count = 0
                    for deal in deals:
                        if deal.magic == MAGIC_NUMBER and deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                            net_pnl = deal.profit + deal.swap + deal.commission
                            payload = {
                                "symbol": deal.symbol, "ticket": deal.ticket,
                                "position_id": deal.position_id, "net_pnl": float(net_pnl),
                                "close_price": deal.price, "reason": deal.reason,
                                "timestamp": deal.time # Server Time
                            }
                            pipe.xadd(CLOSED_TRADE_STREAM, payload, maxlen=1000, approximate=True)
                            published_count += 1
                    if published_count > 0:
                        pipe.execute()
                        log.info(f"📊 Published {published_count} Closed Trades to {CLOSED_TRADE_STREAM}")
            except Exception as e: log.error(f"Closed Trade Monitor Error: {e}")
            time.sleep(5)

    def _sync_positions(self):
        while self.running:
            try:
                with self.mt5_lock:
                    positions = mt5.positions_get()
                    info = mt5.account_info()
                    if info:
                        self.r.hset(CONFIG['redis']['account_info_key'], mapping={
                            "balance": info.balance, "equity": info.equity,
                            "margin": info.margin, 
                            "free_margin": info.margin_free,
                            "updated": time.time()
                        })
                        self.r.set(CONFIG['redis']['risk_keys']['current_equity'], info.equity)
                    if self.ftmo_monitor:
                        self.ftmo_monitor.equity = info.equity
                        if not self.ftmo_monitor.can_trade():
                            log.critical("RISK BREACH DETECTED IN SYNC LOOP. ATTEMPTING LIQUIDATION.")
                            self._close_all_positions()
                    try:
                        hard_deck = float(self.r.get("risk:hard_deck_level") or 0.0)
                        if hard_deck > 0 and info.equity < hard_deck:
                            log.critical(f"💀 HARD DECK BREACHED: Equity {info.equity} < {hard_deck}. LIQUIDATING ALL.")
                            self._close_all_positions()
                    except Exception as e: log.error(f"Hard Deck Check Failed: {e}")
                    
                    pos_list = []
                    if positions:
                        for p in positions:
                            if p.magic == MAGIC_NUMBER:
                                pos_list.append({
                                    "ticket": p.ticket, "symbol": p.symbol,
                                    "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                                    "volume": p.volume, "entry_price": p.price_open,
                                    "profit": p.profit, "sl": p.sl, "tp": p.tp,
                                    "time": p.time, "magic": p.magic, "comment": p.comment 
                                })
                    key = f"{CONFIG['redis']['position_state_key_prefix']}:{MAGIC_NUMBER}"
                    self.r.set(key, json.dumps(pos_list))
            except: pass
            time.sleep(1)

    def _validate_risk_synchronous(self, symbol: str) -> bool:
        try:
            with self.mt5_lock:
                account_info = mt5.account_info()
                if not account_info: return False
                self.ftmo_monitor.equity = account_info.equity
                if not self.ftmo_monitor.can_trade():
                    log.error(f"RISK REJECTION for {symbol}: {self.ftmo_monitor.check_circuit_breakers() if hasattr(self.ftmo_monitor, 'check_circuit_breakers') else 'Circuit Breaker Tripped'}")
                    return False
            return True
        except Exception as e:
            log.error(f"Risk Check Error: {e}")
            return False

    def _midnight_watchman_loop(self):
        log.info("Starting Midnight Watchman (Server Time Authority)...")
        freeze_key = CONFIG['redis']['risk_keys']['midnight_freeze']
        daily_start_key = CONFIG['redis']['risk_keys']['daily_starting_equity']
        hard_deck_key = "risk:hard_deck_level"
        
        last_anchor_id = None
        
        # Load Risk Timezone
        risk_tz_str = CONFIG.get('risk_management', {}).get('risk_timezone', 'Europe/Prague')
        try:
            risk_tz = pytz.timezone(risk_tz_str)
        except:
            risk_tz = pytz.timezone('Europe/Prague')
        
        while self.running and not self.stop_event.is_set():
            # Get Current Server Time (UTC Epoch)
            server_ts = time.time()
            try:
                server_ts += self.exec_engine.broker_time_offset
            except: pass
            
            # Convert to Risk TZ
            dt_utc = datetime.fromtimestamp(server_ts, timezone.utc)
            dt_risk = dt_utc.astimezone(risk_tz)
            
            hour = dt_risk.hour
            minute = dt_risk.minute
            
            # Freeze window: 23:55 to 00:05 in Risk TZ
            if (hour == 23 and minute >= 55) or (hour == 0 and minute <= 5):
                self.r.set(freeze_key, "1")
                
                current_midnight_id = int(server_ts / 86400) # Epoch days as ID
                
                if hour == 0 and minute == 0 and current_midnight_id != last_anchor_id:
                    log.warning(f"⚓ MIDNIGHT ANCHOR ({risk_tz_str}): Capturing Daily Starting Equity...")
                    max_retries = 5
                    for attempt in range(max_retries):
                        with self.mt5_lock:
                            info = mt5.account_info()
                            if info:
                                start_equity = info.equity
                                self.r.set(daily_start_key, start_equity)
                                max_loss_pct = CONFIG['risk_management']['max_daily_loss_pct']
                                loss_limit = start_equity * max_loss_pct
                                hard_deck = start_equity - loss_limit
                                self.r.set(hard_deck_key, hard_deck)
                                log.info(f"⚓ ANCHOR SET: Start {start_equity} | Hard Deck {hard_deck}")
                                last_anchor_id = current_midnight_id
                                break
                        time.sleep(1)
            elif self.r.exists(freeze_key):
                if (hour == 0 and minute > 5) or (hour > 0):
                    self.r.delete(freeze_key)
                    log.info("MIDNIGHT WATCHMAN: Trading Resumed.")
            time.sleep(1)

    def _maintain_time_sync(self):
        while self.running:
            try:
                self.exec_engine._calculate_broker_offset_robust(max_retries=1)
            except:
                pass
            time.sleep(300)

    def _candle_sync_loop(self):
        log.info("Starting Candle Sync Loop...")
        last_processed = {s: 0 for s in SYMBOLS}
        while self.running and not self.stop_event.is_set():
            current_hour_ts = int(time.time() // 3600) * 3600
            for sym in SYMBOLS:
                broker_sym = self.exec_engine.symbol_map.get(sym, sym)
                if last_processed[sym] < current_hour_ts - 3600:
                    start_dt = datetime.utcfromtimestamp(current_hour_ts - 3600)
                    end_dt = datetime.utcfromtimestamp(current_hour_ts)
                    with self.mt5_lock:
                        candles = mt5.copy_rates_range(broker_sym, TIMEFRAME_MT5, start_dt, end_dt)
                        if candles is not None and len(candles) > 0:
                            data_tuples = []
                            for c in candles:
                                ts_iso = pd.to_datetime(c['time'], unit='s', utc=True).isoformat()
                                data_tuples.append((
                                    ts_iso, sym, TARGET_TF_STR,
                                    float(c['open']), float(c['high']), float(c['low']), float(c['close']), int(c['tick_volume'])
                                ))
                            try:
                                with self.db_cursor() as cur:
                                    execute_values(cur, """
                                        INSERT INTO ohlcv (time, symbol, timeframe, open, high, low, close, volume)
                                        VALUES %s ON CONFLICT DO NOTHING
                                    """, data_tuples)
                                last_processed[sym] = current_hour_ts
                                log.info(f"Synced {sym} candles for {start_dt}")
                            except Exception as e: log.error(f"Candle Sync DB Error: {e}")
            time.sleep(60)

    def _pending_order_monitor(self):
        while self.running:
            try:
                if len(self.notified_tickets) > 5000:
                    recent = list(self.notified_tickets)[-1000:]
                    self.notified_tickets = set(recent)
                with self.mt5_lock:
                    orders = mt5.orders_get()
                    if orders:
                        for order in orders:
                            if order.magic == MAGIC_NUMBER:
                                if order.state == mt5.ORDER_STATE_FILLED:
                                    if order.ticket not in self.notified_tickets:
                                        self.r.publish("order_filled_channel", json.dumps({
                                            'ticket': order.ticket, 'symbol': order.symbol, 'type': 'FILLED'
                                        }))
                                        self.notified_tickets.add(order.ticket)
                                if time.time() - order.time_setup > 60:
                                    log.info(f"Cancelling Zombie Order {order.ticket} (Pending > 60s)")
                                    req = {"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket}
                                    mt5.order_send(req)
            except Exception as e: log.error(f"Pending Monitor Error: {e}")
            time.sleep(1)

    def run(self):
        threads = [
            threading.Thread(target=self._tick_stream_loop, daemon=True),
            threading.Thread(target=self._trade_listener, daemon=True),
            threading.Thread(target=self._sync_positions, daemon=True),
            threading.Thread(target=self._candle_sync_loop, daemon=True),
            threading.Thread(target=self._midnight_watchman_loop, daemon=True),
            threading.Thread(target=self._pending_order_monitor, daemon=True),
            threading.Thread(target=self._closed_trade_monitor, daemon=True),
            threading.Thread(target=self._maintain_time_sync, daemon=True) 
        ]
        for t in threads: t.start()
        log.info(f"{LogSymbols.ONLINE} Windows Producer Running. Risk State: {self.ftmo_monitor.starting_equity_of_day}")
        try:
            while self.running:
                try:
                    msg_id, data = self.execution_queue.get(timeout=1)
                    if "action" in data:
                        # Only run risk check for trade entries (BUY/SELL or 1/5)
                        act = data["action"]
                        is_entry = (act in ["BUY", "SELL"]) or (str(act) in ["1", "5"])
                        
                        if is_entry:
                            if not self._validate_risk_synchronous(data["symbol"]):
                                log.warning(f"Trade blocked by Sync Risk Check: {data['symbol']}")
                                self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                                continue
                    
                    self._process_trade_signal_sync(msg_id, data)
                    
                except queue.Empty: pass
        except KeyboardInterrupt: log.info("Stopping...")
        finally:
            self.running = False
            self.stop_event.set()
            self.executor.shutdown(wait=False)
            with self.mt5_lock: mt5.shutdown()

if __name__ == "__main__":
    try:
        producer = HybridProducer()
        producer.run()
    except Exception as e:
        log.critical(f"FATAL: {e}")
        input("Press Enter to exit...")