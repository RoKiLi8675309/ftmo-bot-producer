# =============================================================================
# FILENAME: windows_producer.py
# ENVIRONMENT: Windows (Python 3.9) - MT5 Host
# DEPENDENCIES: shared package, MetaTrader5, psycopg2, psutil
# DESCRIPTION:
# The Gateway to the Market.
# 1. Downloads Historical Data -> Postgres.
# 2. Streams Live Ticks -> Redis.
# 3. Executes Trades <- Redis (Limit Orders).
# 4. Publishes Closed Trades -> Redis (Critical for V10.0 Circuit Breaker).
#
# PHOENIX V12.4 UPDATE (SNIPER MODE - AGGRESSOR PROTOCOL):
# - SYNC: Subscription optimized for High-Vol pairs (GBPAUD, EURJPY).
# - RISK: "Midnight Anchor" captures 00:00 Server Time Equity for Daily Limits.
# =============================================================================
import os
import sys
import time
import json
import logging
import threading
import psutil
import signal
import math
import queue
import gc
import uuid
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# --- PATH SAFETY FIX ---
# Ensure the current directory is in sys.path so 'shared' can be imported
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

# --- PROJECT IMPORTS (MODULAR) ---
try:
    from shared import (
        CONFIG,
        setup_logging, LogSymbols,
        get_redis_connection,
        FTMORiskMonitor, RiskManager, TradeContext, SessionGuard,
        NewsEventMonitor, FTMOComplianceGuard,
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
# Ensure we also subscribe to auxiliaries for conversion
AUX_SYMBOLS = CONFIG['trading'].get('auxiliary_symbols', [])
ALL_MONITORED_SYMBOLS = list(set(SYMBOLS + AUX_SYMBOLS))

STREAM_KEY = CONFIG['redis']['price_data_stream']
TRADE_REQUEST_STREAM = CONFIG['redis']['trade_request_stream']
CLOSED_TRADE_STREAM = CONFIG['redis'].get('closed_trade_stream_key', 'stream:closed_trades')
MAGIC_NUMBER = CONFIG['trading']['magic_number']
MIDNIGHT_BUFFER_MINUTES = 30
KILL_SWITCH_FILE = "kill_switch.lock"

# Execution Settings (Limit Order Offset)
LIMIT_OFFSET_PIPS = CONFIG['trading'].get('limit_order_offset_pips', 0.2)

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
    """
    Monitors latency statistics to dynamically adjust trade signal Time-To-Live (TTL).
    """
    def __init__(self, base_ttl=3.0, max_ttl=5.0, alpha=0.1):
        self.base_ttl = base_ttl
        self.max_ttl = max_ttl
        self.alpha = alpha
        
        # State
        self.avg_latency = 0.5 # Assume 500ms start
        self.latency_var = 0.0 # Variance
        self.current_ttl = base_ttl
        self.lock = threading.Lock()
        
    def update(self, latency: float):
        with self.lock:
            delta = latency - self.avg_latency
            self.avg_latency += self.alpha * delta
            self.latency_var = (1 - self.alpha) * self.latency_var + self.alpha * (delta ** 2)
            
            jitter = math.sqrt(self.latency_var)
            if self.avg_latency < 1.0 and jitter < 0.5:
                self.current_ttl = self.max_ttl
            else:
                self.current_ttl = self.base_ttl

    def get_ttl(self) -> float:
        return self.current_ttl

class MT5ExecutionEngine:
    """
    Robust Execution Engine.
    Handles Idempotency, Retries, Broker Constraints, and Metrics.
    """
    def __init__(self, redis_client, lock: threading.RLock, risk_monitor: FTMORiskMonitor):
        self.lock = lock
        self.default_deviation = CONFIG['trading'].get('slippage', 5)
        self.magic_number = CONFIG['trading']['magic_number']
        self.r = redis_client
        self.risk_monitor = risk_monitor
        self.broker_time_offset = 0.0
        
        if not self._calculate_broker_offset_robust():
            log.critical("CRITICAL: Timezone Sync Failed. Aborting startup.")
            raise RuntimeError("Timezone Sync Failed")

    def _calculate_broker_offset_robust(self, max_retries: int = 10) -> bool:
        log.info("Calculating Broker-Local Time Offset...")
        ref_symbol = SYMBOLS[0] if SYMBOLS else "EURUSD"
        for attempt in range(max_retries):
            with self.lock:
                if not mt5.symbol_select(ref_symbol, True):
                    time.sleep(1)
                    continue
                tick = mt5.symbol_info_tick(ref_symbol)
                if tick:
                    server_ts = tick.time
                    local_ts = datetime.now(timezone.utc).timestamp()
                    self.broker_time_offset = server_ts - local_ts
                    try:
                        self.r.set("producer:broker_time_offset", self.broker_time_offset)
                    except: pass
                    log.info(f"TIMEZONE AUDIT: Broker Offset: {self.broker_time_offset:.2f}s")
                    return True
            time.sleep(1)
        return False

    def _get_symbol_info(self, symbol: str):
        with self.lock:
            info = mt5.symbol_info(symbol)
            if info and not info.visible:
                mt5.symbol_select(symbol, True)
                info = mt5.symbol_info(symbol)
            return info

    def _check_idempotency(self, symbol: str, comment: str, lookback_sec: int = 45) -> Optional[Dict[str, Any]]:
        now_broker = time.time() + self.broker_time_offset
        cutoff_msc = (now_broker - lookback_sec) * 1000
        with self.lock:
            positions = mt5.positions_get(symbol=symbol)
            if positions:
                for pos in positions:
                    if pos.comment == comment and pos.time_msc > cutoff_msc:
                        log.warning(f"IDEMPOTENCY: Found existing position {pos.ticket}.")
                        return {"retcode": mt5.TRADE_RETCODE_DONE, "order": pos.ticket, "price": pos.price_open}
            
            orders = mt5.orders_get(symbol=symbol)
            if orders:
                for order in orders:
                    if order.comment == comment and order.time_setup_msc > cutoff_msc:
                        log.warning(f"IDEMPOTENCY: Found existing pending order {order.ticket}.")
                        return {"retcode": mt5.TRADE_RETCODE_PLACED, "order": order.ticket}
        return None

    def execute_trade(self, request: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        symbol = request.get("symbol")
        if not symbol: return None
        
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info: return None
        
        try:
            with self.lock:
                tick = mt5.symbol_info_tick(symbol)
                if not tick: return None
            
            pip_size = symbol_info.point * 10 if symbol_info.digits == 3 or symbol_info.digits == 5 else symbol_info.point
            
            if request["action"] == mt5.TRADE_ACTION_DEAL:
                offset_val = LIMIT_OFFSET_PIPS * pip_size
                
                if request["type"] == mt5.ORDER_TYPE_BUY:
                    limit_price = tick.bid - offset_val
                    request["type"] = mt5.ORDER_TYPE_BUY_LIMIT
                    request["price"] = limit_price
                    request["action"] = mt5.TRADE_ACTION_PENDING
                elif request["type"] == mt5.ORDER_TYPE_SELL:
                    limit_price = tick.ask + offset_val
                    request["type"] = mt5.ORDER_TYPE_SELL_LIMIT
                    request["price"] = limit_price
                    request["action"] = mt5.TRADE_ACTION_PENDING
                
                request["type_time"] = mt5.ORDER_TIME_GTC
            
            if "price" in request:
                request["price"] = PrecisionGuard.normalize_price(request["price"], symbol, symbol_info)
            
            raw_vol = float(request["volume"])
            vol_step = symbol_info.volume_step
            if vol_step > 0:
                steps = round(raw_vol / vol_step)
                request["volume"] = steps * vol_step
            request["volume"] = max(symbol_info.volume_min, min(request["volume"], symbol_info.volume_max))
            request["volume"] = round(request["volume"], 2)

            if "sl" in request and float(request["sl"]) > 0:
                request["sl"] = PrecisionGuard.normalize_price(float(request["sl"]), symbol, symbol_info)
            if "tp" in request and float(request["tp"]) > 0:
                request["tp"] = PrecisionGuard.normalize_price(float(request["tp"]), symbol, symbol_info)
            
            filling = symbol_info.filling_mode
            if filling & mt5.SYMBOL_FILLING_FOK:
                request["type_filling"] = mt5.ORDER_FILLING_FOK
            elif filling & mt5.SYMBOL_FILLING_IOC:
                request["type_filling"] = mt5.ORDER_FILLING_IOC
            else:
                request["type_filling"] = mt5.ORDER_FILLING_RETURN
                
        except ValueError as e:
            log.error(f"Sanitization error for {symbol}: {e}")
            return None

        for attempt in range(max_retries):
            with self.lock:
                result = mt5.order_send(request)
            
            if result is None:
                ghost = self._check_idempotency(symbol, request.get("comment", ""))
                if ghost: return ghost
                time.sleep(0.5)
                continue
                
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                log.info(f"EXECUTION SUCCESS: {symbol} Ticket: {result.order}")
                return result._asdict()
            elif result.retcode == mt5.TRADE_RETCODE_PLACED:
                log.info(f"LIMIT ORDER PLACED: {symbol} Ticket: {result.order} @ {request['price']}")
                return result._asdict()
            elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_CONNECTION, mt5.TRADE_RETCODE_PRICE_OFF]:
                try: self.r.incr("broker:metric:requotes")
                except: pass
                
                log.warning(f"Recoverable Error ({result.retcode}). Retrying...")
                time.sleep(0.5 * (2 ** attempt))
                
                if request["action"] == mt5.TRADE_ACTION_PENDING:
                    with self.lock:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            offset_val = LIMIT_OFFSET_PIPS * pip_size
                            if request["type"] == mt5.ORDER_TYPE_BUY_LIMIT:
                                request["price"] = tick.bid - offset_val
                            elif request["type"] == mt5.ORDER_TYPE_SELL_LIMIT:
                                request["price"] = tick.ask + offset_val
                    continue
            else:
                log.error(f"EXECUTION FAILURE: {symbol} Retcode: {result.retcode} ({result.comment})")
                try:
                    self.r.publish("order_failed_channel", json.dumps({"symbol": symbol, "reason": result.comment}))
                except: pass
                break
        
        return None

    def close_position(self, position_id: int, symbol: str, volume: float, pos_type: int) -> Optional[Any]:
        with self.lock:
            tick = mt5.symbol_info_tick(symbol)
            if not tick: return None
            
            price = tick.bid if pos_type == mt5.ORDER_TYPE_BUY else tick.ask
            trade_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": trade_type,
                "position": position_id,
                "price": price,
                "deviation": self.default_deviation,
                "magic": self.magic_number,
                "comment": "Algo Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            result = mt5.order_send(request)
            return result

class HybridProducer:
    def __init__(self):
        self.running = True
        self.stop_event = threading.Event()
        self.r = get_redis_connection(
            host=CONFIG['redis']['host'],
            port=CONFIG['redis']['port'],
            db=0,
            decode_responses=True
        )
        self.mt5_lock = threading.RLock()
        self.execution_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ExecWorker")
        self.db_dsn = CONFIG['postgres']['dsn']
        self._connect_db_with_retry()
        
        initial_bal = CONFIG.get('env', {}).get('initial_balance', 100000.0)
        self.ftmo_monitor = FTMORiskMonitor(
            initial_balance=initial_bal,
            max_daily_loss_pct=CONFIG['risk_management']['max_daily_loss_pct'],
            redis_client=self.r
        )
        
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
        
        self.last_prices = {s: 0.0 for s in SYMBOLS}
        self.last_deal_scan_time = datetime.now() - timedelta(minutes=1)
        
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
            if path and os.path.exists(path):
                init_params['path'] = path
            
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
            
            # Subscribe to ALL configured symbols (Target + Aux)
            for sym in ALL_MONITORED_SYMBOLS:
                if not mt5.symbol_select(sym, True):
                    log.error(f"Failed to select symbol {sym}")
                else:
                    log.info(f"Subscribed to {sym}")

    def _reconstruct_risk_state_from_history(self):
        log.info("Reconstructing Risk State (FTMO Compliance)...")
        retry_count = 0
        max_retries = 5
        
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
                    
                    now = datetime.now()
                    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    deals = mt5.history_deals_get(midnight, now)
                    
                    if deals is None:
                        log.error(f"MT5 History API returned None (Audit Fail). Retry {retry_count+1}...")
                        retry_count += 1
                        time.sleep(2)
                        continue
                    
                    realized_pnl_today = 0.0
                    for d in deals:
                        if d.magic == MAGIC_NUMBER:
                            realized_pnl_today += (d.profit + d.swap + d.commission)
                    
                    # Re-calculate Starting Equity for today
                    calculated_start = current_balance - realized_pnl_today
                    self.ftmo_monitor.starting_equity_of_day = calculated_start
                    self.r.set(CONFIG['redis']['risk_keys']['daily_starting_equity'], calculated_start)
                    self.r.set("bot:account_size", self.ftmo_monitor.initial_balance) 
                    
                    # V10.0: Set Hard Deck Level immediately
                    loss_limit = calculated_start * CONFIG['risk_management']['max_daily_loss_pct']
                    hard_deck = calculated_start - loss_limit
                    self.r.set("risk:hard_deck_level", hard_deck)
                    
                    log.info(f"{LogSymbols.SUCCESS} RISK STATE VERIFIED: Start Equity: {calculated_start:.2f} | Hard Deck: {hard_deck:.2f} | Account: {self.ftmo_monitor.initial_balance}")
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
                        with self.mt5_lock:
                            rates = mt5.copy_rates_range(sym, TIMEFRAME_MT5, current_start, current_end)
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
        # Start with all symbols explicitly configured
        monitored = set(ALL_MONITORED_SYMBOLS)
        with self.mt5_lock:
            account_info = mt5.account_info()
            if not account_info: return monitored
            acc_ccy = account_info.currency
            
            for sym in SYMBOLS:
                info = mt5.symbol_info(sym)
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
            if last_price == 0:
                return current_vol / 2.0, current_vol / 2.0
                
            if current_price > last_price:
                return float(current_vol), 0.0
            elif current_price < last_price:
                return 0.0, float(current_vol)
            else:
                half_vol = float(current_vol) / 2.0
                return half_vol, half_vol
        except Exception:
            return 0.0, 0.0

    def _tick_stream_loop(self):
        log.info("Starting Tick Stream...")
        interval = CONFIG['producer']['tick_interval_seconds']
        redis_failures = 0
        
        while self.running and not self.stop_event.is_set():
            if os.path.exists(KILL_SWITCH_FILE):
                log.critical("💀 KILL SWITCH DETECTED. SHUTTING DOWN IMMEDIATELY.")
                with self.mt5_lock: mt5.shutdown()
                sys.exit(0)
                
            start = time.time()
            try:
                # Update Context periodically
                if time.time() - self.last_context_update > 60:
                    self._update_d1_context()
                    self._update_h4_context()
                    self.last_context_update = time.time()
                
                pipe = self.r.pipeline()
                for sym in self.monitored_symbols:
                    with self.mt5_lock:
                        tick = mt5.symbol_info_tick(sym)
                        if not tick: continue
                        
                        sym_info = mt5.symbol_info(sym)
                        digits = sym_info.digits if sym_info else 5
                        bid = round(tick.bid, digits)
                        ask = round(tick.ask, digits)
                        current_vol = float(tick.volume_real if tick.volume_real > 0 else tick.volume)
                        price_now = tick.last if tick.last > 0 else (bid + ask) / 2.0
                        
                        bid_vol, ask_vol = self._estimate_flow_volumes(sym, current_vol, price_now)
                        self.last_prices[sym] = price_now
                        
                        self.cluster_engine.update_correlations(pd.DataFrame())
                        utc_ts = int(tick.time_msc) - int(self.exec_engine.broker_time_offset * 1000)
                        
                        payload = {
                            "symbol": sym, "time": utc_ts,
                            "bid": bid,
                            "ask": ask,
                            "price": price_now,
                            "volume": current_vol,
                            "bid_vol": float(bid_vol),
                            "ask_vol": float(ask_vol),
                            "ctx_d1": json.dumps(self.d1_cache.get(sym, {})),
                            "ctx_h4": json.dumps(self.h4_cache.get(sym, {}))
                        }
                        
                        if sym in SYMBOLS: pipe.xadd(STREAM_KEY, payload, maxlen=10000, approximate=True)
                        key = self.monitored_price_keys.get(sym, f"price:{sym}")
                        pipe.hset(key, mapping={"bid": payload["bid"], "ask": payload["ask"], "time": int(utc_ts/1000)})
                
                pipe.execute()
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
        """
        V10.0 UPDATE: Precision EMA calculation for Trend Filter.
        Fetches 500 bars to ensure EMA 200 convergence.
        """
        for sym in SYMBOLS:
            with self.mt5_lock:
                # Fetch 500 bars for accurate tail (200 is insufficient for startup)
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, 500)
                if rates is not None and len(rates) > 200:
                    r = rates[-1]
                    
                    # Convert to Series
                    closes = pd.Series([x['close'] for x in rates])
                    
                    # V10.0: Standard Pandas EWM (span=200) for mathematical consistency
                    ema_series = closes.ewm(span=200, adjust=False).mean()
                    ema = ema_series.iloc[-1]
                    
                    self.d1_cache[sym] = {
                        'open': float(r['open']), 'high': float(r['high']),
                        'low': float(r['low']), 'close': float(r['close']),
                        'ema200': float(ema)
                    }

    def _update_h4_context(self):
        """
        V10.0 UPDATE: Standard RSI 14 (Wilder's Smoothing).
        """
        for sym in SYMBOLS:
            with self.mt5_lock:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_H4, 1, 100) # Fetch more for RSI smoothing
                if rates is not None and len(rates) > 14:
                    r = rates[-1]
                    
                    # RSI Calculation (Standard Wilder's)
                    df = pd.DataFrame(rates)
                    delta = df['close'].diff()
                    
                    # Wilder's Smoothing: alpha = 1/N (com = N-1)
                    # For RSI 14, com = 13
                    gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                    loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                    
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    if np.isnan(current_rsi): current_rsi = 50.0
                    
                    self.h4_cache[sym] = {
                        'open': float(r['open']),
                        'high': float(r['high']),
                        'low': float(r['low']),
                        'close': float(r['close']),
                        'rsi': float(current_rsi)
                    }

    def _trade_listener(self):
        log.info("Starting Trade Listener...")
        redis_failures = 0
        try: self.r.xgroup_create(TRADE_REQUEST_STREAM, "execution_group", id='0', mkstream=True)
        except: pass
        
        while self.running:
            try:
                if int(time.time()) % 60 == 0:
                    self.r.xtrim(TRADE_REQUEST_STREAM, maxlen=1000, approximate=True)
                entries = self.r.xreadgroup("execution_group", "producer_main", {TRADE_REQUEST_STREAM: '>'}, count=5, block=1000)
                redis_failures = 0 
                
                if entries:
                    for stream, messages in entries:
                        for msg_id, data in messages:
                            self.execution_queue.put((msg_id, data))
            
            except (redis.ConnectionError, redis.TimeoutError) as e:
                redis_failures += 1
                sleep_time = min(2 ** redis_failures, 30)
                log.error(f"Redis Listener Disconnected. Retrying in {sleep_time}s... ({e})")
                time.sleep(sleep_time)
            except Exception as e:
                log.error(f"Trade Listener Error: {e}")
                time.sleep(1)

    def _async_trade_worker(self, msg_id, data):
        try:
            symbol = data['symbol']
            action = data['action']
            uuid_val = data.get('uuid')
            
            request_ts_str = data.get('timestamp')
            
            if request_ts_str:
                try:
                    request_ts = float(request_ts_str)
                    now_ts = time.time()
                    latency = now_ts - request_ts
                    
                    self.ttl_manager.update(latency)
                    current_ttl = self.ttl_manager.get_ttl()
                    
                    if latency > current_ttl:
                        log.error(f"🧟 ZOMBIE TRADE DROPPED: {symbol} Lag: {latency:.2f}s > {current_ttl:.1f}s")
                        self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                        return
                except ValueError:
                    log.error(f"Invalid timestamp: {request_ts_str}")

            if uuid_val:
                if self.r.sismember("processed_signals", uuid_val):
                    self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                    return
                self.r.sadd("processed_signals", uuid_val)
                self.r.expire("processed_signals", 3600)

            if action in ["BUY", "SELL"]:
                with self.mt5_lock:
                    info = mt5.account_info()
                    if info:
                        self.ftmo_monitor.equity = info.equity
                        self.ftmo_monitor._check_constraints(0.0)
                        if not self.ftmo_monitor.can_trade():
                            log.warning("Trade blocked by Risk Monitor.")
                            self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                            return
                self.exec_engine.execute_trade({
                    "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
                    "volume": float(data['volume']),
                    "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
                    "sl": float(data.get("stop_loss", 0)), "tp": float(data.get("take_profit", 0)),
                    "magic": MAGIC_NUMBER, "comment": data.get("comment", "Algo")
                })
            elif action == "MODIFY":
                self.exec_engine.execute_trade({
                    "action": mt5.TRADE_ACTION_SLTP, "position": int(data['ticket']),
                    "sl": float(data.get("sl", 0)), "tp": float(data.get("tp", 0))
                })
            elif action == "CLOSE_ALL":
                self._close_all_positions(symbol)
                
            self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
        except Exception as e:
            log.error(f"Trade Execution Error: {e}")

    def _close_all_positions(self, symbol: str = None):
        with self.mt5_lock:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if not positions: return
            for p in positions:
                if p.magic != MAGIC_NUMBER: continue
                self.exec_engine.close_position(p.ticket, p.symbol, p.volume, p.type)

    def _closed_trade_monitor(self):
        """
        Polls MT5 for closed trades and publishes them to Redis Stream.
        V10.0: Critical for Engine's Daily Loss Limit.
        """
        log.info("Starting Closed Trade Monitor (SQN Feed)...")
        while self.running and not self.stop_event.is_set():
            try:
                now = datetime.now()
                with self.mt5_lock:
                    deals = mt5.history_deals_get(self.last_deal_scan_time, now)
                
                self.last_deal_scan_time = now
                
                if deals:
                    pipe = self.r.pipeline()
                    published_count = 0
                    
                    for deal in deals:
                        if deal.magic == MAGIC_NUMBER and deal.entry in [mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_INOUT]:
                            net_pnl = deal.profit + deal.swap + deal.commission
                            
                            payload = {
                                "symbol": deal.symbol,
                                "ticket": deal.ticket,
                                "position_id": deal.position_id,
                                "net_pnl": float(net_pnl),
                                "close_price": deal.price,
                                "reason": deal.reason,
                                "timestamp": deal.time
                            }
                            pipe.xadd(CLOSED_TRADE_STREAM, payload, maxlen=1000, approximate=True)
                            published_count += 1
                            
                    if published_count > 0:
                        pipe.execute()
                        log.info(f"📊 Published {published_count} Closed Trades to {CLOSED_TRADE_STREAM}")
            
            except Exception as e:
                log.error(f"Closed Trade Monitor Error: {e}")
                
            time.sleep(5)

    def _sync_positions(self):
        """
        Syncs positions and Enforces HARD DECK (Daily Limit).
        V11.1: Adds 'time' field for Engine Time-Stop enforcement.
        """
        while self.running:
            try:
                with self.mt5_lock:
                    positions = mt5.positions_get()
                    info = mt5.account_info()
                    
                    if info:
                        self.r.hset(CONFIG['redis']['account_info_key'], mapping={
                            "balance": info.balance, "equity": info.equity,
                            "margin": info.margin, "free_margin": info.margin_free, "updated": time.time()
                        })
                        self.r.set(CONFIG['redis']['risk_keys']['current_equity'], info.equity)
                    
                    if self.ftmo_monitor:
                        self.ftmo_monitor.equity = info.equity
                        if not self.ftmo_monitor.can_trade():
                            log.critical("RISK BREACH DETECTED IN SYNC LOOP. ATTEMPTING LIQUIDATION.")
                            self._close_all_positions()
                    
                    # --- V10.0 HARD DECK ENFORCEMENT ---
                    try:
                        hard_deck = float(self.r.get("risk:hard_deck_level") or 0.0)
                        if hard_deck > 0 and info.equity < hard_deck:
                            log.critical(f"💀 HARD DECK BREACHED: Equity {info.equity} < {hard_deck}. LIQUIDATING ALL.")
                            self._close_all_positions()
                    except Exception as e:
                        log.error(f"Hard Deck Check Failed: {e}")
                    # -----------------------------------

                    pos_list = []
                    if positions:
                        for p in positions:
                            if p.magic == MAGIC_NUMBER:
                                pos_list.append({
                                    "ticket": p.ticket, "symbol": p.symbol,
                                    "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                                    "volume": p.volume, "entry_price": p.price_open,
                                    "profit": p.profit, "sl": p.sl, "tp": p.tp,
                                    "time": p.time, "magic": p.magic, "comment": p.comment # V11.1 Update
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
                self.ftmo_monitor._check_constraints(0.0)
                if not self.ftmo_monitor.can_trade():
                    log.error(f"RISK REJECTION: {self.ftmo_monitor.check_circuit_breakers()}")
                    return False
            return True
        except Exception as e:
            log.error(f"Risk Check Error: {e}")
            return False

    def _midnight_watchman_loop(self):
        """
        V10.0: Midnight Anchor Protocol.
        Captures Server 00:00 Equity accurately to establish daily loss limits.
        """
        log.info("Starting Midnight Watchman (Anchor Protocol)...")
        freeze_key = CONFIG['redis']['risk_keys']['midnight_freeze']
        daily_start_key = CONFIG['redis']['risk_keys']['daily_starting_equity']
        hard_deck_key = "risk:hard_deck_level"
        
        # Use config timezone or default to Prague (FTMO server time)
        tz_name = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        risk_tz = pytz.timezone(tz_name)
        
        last_anchor_date = None
        
        while self.running and not self.stop_event.is_set():
            # Get current server time
            now_utc = datetime.now(timezone.utc)
            now_server = now_utc.astimezone(risk_tz)
            
            # --- PHASE 1: FREEZE (23:55 - 00:05) ---
            if (now_server.hour == 23 and now_server.minute >= 55) or (now_server.hour == 0 and now_server.minute <= 5):
                self.r.set(freeze_key, "1")
                
                # --- PHASE 2: ANCHOR SNAPSHOT (00:00:01) ---
                # We do this once per day, right after midnight
                if now_server.hour == 0 and now_server.minute == 0 and now_server.date() != last_anchor_date:
                    log.warning("⚓ MIDNIGHT ANCHOR: Capturing Daily Starting Equity...")
                    
                    max_retries = 5
                    for attempt in range(max_retries):
                        with self.mt5_lock:
                            info = mt5.account_info()
                            if info:
                                start_equity = info.equity
                                
                                # 1. Set Daily Start
                                self.r.set(daily_start_key, start_equity)
                                
                                # 2. Calculate Hard Deck
                                max_loss_pct = CONFIG['risk_management']['max_daily_loss_pct']
                                loss_limit = start_equity * max_loss_pct
                                hard_deck = start_equity - loss_limit
                                self.r.set(hard_deck_key, hard_deck)
                                
                                log.info(f"⚓ ANCHOR SET: Start {start_equity} | Hard Deck {hard_deck}")
                                last_anchor_date = now_server.date()
                                break
                        time.sleep(1)
            
            # --- PHASE 3: UNFREEZE ---
            elif self.r.exists(freeze_key):
                if (now_server.hour == 0 and now_server.minute > 5) or (now_server.hour > 0):
                    self.r.delete(freeze_key)
                    log.info("MIDNIGHT WATCHMAN: Trading Resumed.")
            
            time.sleep(1)

    def _candle_sync_loop(self):
        log.info("Starting Candle Sync Loop...")
        last_processed = {s: 0 for s in SYMBOLS}
        while self.running and not self.stop_event.is_set():
            current_hour_ts = int(time.time() // 3600) * 3600
            for sym in SYMBOLS:
                if last_processed[sym] < current_hour_ts - 3600:
                    start_dt = datetime.utcfromtimestamp(current_hour_ts - 3600)
                    end_dt = datetime.utcfromtimestamp(current_hour_ts)
                    with self.mt5_lock:
                        candles = mt5.copy_rates_range(sym, TIMEFRAME_MT5, start_dt, end_dt)
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
                            except Exception as e:
                                log.error(f"Candle Sync DB Error: {e}")
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
            except Exception as e:
                log.error(f"Pending Monitor Error: {e}")
            time.sleep(1)

    def run(self):
        threads = [
            threading.Thread(target=self._tick_stream_loop, daemon=True),
            threading.Thread(target=self._trade_listener, daemon=True),
            threading.Thread(target=self._sync_positions, daemon=True),
            threading.Thread(target=self._candle_sync_loop, daemon=True),
            threading.Thread(target=self._midnight_watchman_loop, daemon=True),
            threading.Thread(target=self._pending_order_monitor, daemon=True),
            threading.Thread(target=self._closed_trade_monitor, daemon=True)
        ]
        for t in threads: t.start()
        
        log.info(f"{LogSymbols.ONLINE} Windows Producer Running. Risk State: {self.ftmo_monitor.starting_equity_of_day}")
        
        try:
            while self.running:
                try:
                    msg_id, data = self.execution_queue.get(timeout=1)
                    if "action" in data and data["action"] in ["BUY", "SELL"]:
                        if not self._validate_risk_synchronous(data["symbol"]):
                            log.warning(f"Trade blocked by Sync Risk Check: {data['symbol']}")
                            self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                            continue
                    self.executor.submit(self._async_trade_worker, msg_id, data)
                except queue.Empty: pass
        except KeyboardInterrupt:
            log.info("Stopping...")
        finally:
            self.running = False
            self.stop_event.set()
            self.executor.shutdown(wait=False)
            with self.mt5_lock:
                mt5.shutdown()

if __name__ == "__main__":
    try:
        producer = HybridProducer()
        producer.run()
    except Exception as e:
        log.critical(f"FATAL: {e}")
        input("Press Enter to exit...")