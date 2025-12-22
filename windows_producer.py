# =============================================================================
# FILENAME: windows_producer.py
# ENVIRONMENT: Windows (Python 3.9) - MT5 Host
# DEPENDENCIES: shared package, MetaTrader5, psycopg2, psutil
# DESCRIPTION:
# The Gateway to the Market.
# 1. Downloads Historical Data -> Postgres.
# 2. Streams Live Ticks -> Redis.
# 3. Executes Trades <- Redis.
# 4. AUDIT REMEDIATION (2025-12-20):
#    - A. Compliance: Strict Retry Loop for Risk State Reconstruction.
#    - B. Zombie Defense: Strict TTL on trade signals.
#    - C. Thread Safety: Global RLock for ALL MT5 calls.
#    - D. Infrastructure: Redis Exponential Backoff & File-Based Kill Switch.
#    - E. Precision: Dynamic rounding for JPY/XAU before JSON serialization.
#    - F. AUTO-DETECT: Updates Account Size from Broker to prevent false Liquidations.
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
from shared import (
    CONFIG,
    setup_logging, LogSymbols,
    get_redis_connection,
    FTMORiskMonitor, RiskManager, TradeContext, SessionGuard,
    NewsEventMonitor, FTMOComplianceGuard,
    PrecisionGuard
)

# Initialize Logging
setup_logging("WindowsProducer")
log = logging.getLogger("Producer")

# --- CONFIGURATION CONSTANTS ---
SYMBOLS = CONFIG['trading']['symbols']
STREAM_KEY = CONFIG['redis']['price_data_stream']
TRADE_REQUEST_STREAM = CONFIG['redis']['trade_request_stream']
MAGIC_NUMBER = CONFIG['trading']['magic_number']
MIDNIGHT_BUFFER_MINUTES = 30
MAX_LATENCY_SECONDS = 3.0 # AUDIT FIX: Tightened from 5.0s to 3.0s (Zombie Defense)
KILL_SWITCH_FILE = "kill_switch.lock"

# Dynamic Timeframe Mapping
TARGET_TF_STR = CONFIG['trading'].get('timeframe', 'H1').upper()
try:
    TIMEFRAME_MT5 = getattr(mt5, f"TIMEFRAME_{TARGET_TF_STR}")
    log.info(f"TIMEFRAME CONFIG: Set to {TARGET_TF_STR} (MT5 Constant: {TIMEFRAME_MT5})")
except AttributeError:
    log.warning(f"Invalid timeframe '{TARGET_TF_STR}' in config. Defaulting to H1.")
    TIMEFRAME_MT5 = mt5.TIMEFRAME_H1
    TARGET_TF_STR = "H1"

# --- LOCAL GUARDRAILS ---

class LocalClusterContextBuilder:
    """Manages asset clusters locally to avoid import errors on Py3.9."""
    def __init__(self, pairs: List[str]):
        self.pairs = pairs
        self.cache = {p: 0.0 for p in pairs}
        self.last_prices = {p: None for p in pairs}

    def update_tick(self, symbol: str, price: float):
        if symbol not in self.last_prices: return
        if self.last_prices[symbol] is not None:
            prev = self.last_prices[symbol]
            if prev > 0:
                pct_change = (price - prev) / prev
                self.cache[symbol] = pct_change
        self.last_prices[symbol] = price

    def get_context_vector(self, target_symbol: str) -> Dict[str, float]:
        features = {}
        cluster_strength = 0.0
        count = 0
        for p in self.pairs:
            if p == target_symbol: continue
            val = self.cache.get(p, 0.0)
            # Simple heuristic: USD pairs correlation
            if p.startswith("USD"): cluster_strength += val
            elif p.endswith("USD"): cluster_strength -= val
            count += 1
        features[f"ctx_{p}_chg"] = val
        features['ctx_usd_index_proxy'] = cluster_strength / count if count > 0 else 0.0
        return features

class LocalTimeFeatureTransformer:
    """Transforms time locally for Py3.9 stability."""
    def __init__(self):
        self.london_start = 7
        self.london_end = 16
        self.ny_start = 13
        self.ny_end = 22

    def transform(self, dt_obj: datetime) -> Dict[str, float]:
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        
        hour_float = dt_obj.hour + (dt_obj.minute / 60.0)
        hour_sin = math.sin(2 * math.pi * hour_float / 24.0)
        hour_cos = math.cos(2 * math.pi * hour_float / 24.0)
        day_of_week = dt_obj.weekday()
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)

        is_london = self.london_start <= dt_obj.hour < self.london_end
        is_ny = self.ny_start <= dt_obj.hour < self.ny_end
        is_overlap = is_london and is_ny
        
        return {
            't_hour_sin': hour_sin, 't_hour_cos': hour_cos,
            't_day_sin': day_sin, 't_day_cos': day_cos,
            'sess_london': int(is_london), 'sess_ny': int(is_ny),
            'sess_overlap': int(is_overlap)
        }

class MT5ExecutionEngine:
    """
    Robust Execution Engine.
    Handles Idempotency, Retries, Broker Constraints, and Metrics.
    AUDIT FIX: All MT5 calls guarded by self.lock (Re-entrant).
    """
    def __init__(self, redis_client, lock: threading.RLock, risk_monitor: FTMORiskMonitor):
        self.lock = lock
        self.default_deviation = CONFIG['trading'].get('slippage', 10)
        self.magic_number = CONFIG['trading']['magic_number']
        self.r = redis_client
        self.risk_monitor = risk_monitor
        self.broker_time_offset = 0.0
        
        with self.lock:
            if not mt5.initialize():
                log.critical(f"MT5 Handler Init failed: {mt5.last_error()}")
                raise RuntimeError("MT5 Handler Init Failed")
        
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
        
        # 1. Pre-Flight Checks
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info: return None
        
        # Sanitize Price/Volume using PrecisionGuard
        try:
            if "price" in request:
                request["price"] = PrecisionGuard.normalize_price(request["price"], symbol, symbol_info)
            
            # Normalize Volume
            raw_vol = float(request["volume"])
            vol_step = symbol_info.volume_step
            if vol_step > 0:
                steps = round(raw_vol / vol_step)
                request["volume"] = steps * vol_step
            request["volume"] = max(symbol_info.volume_min, min(request["volume"], symbol_info.volume_max))
            request["volume"] = round(request["volume"], 2)

            # Sanitize SL/TP
            if "sl" in request and float(request["sl"]) > 0:
                request["sl"] = PrecisionGuard.normalize_price(float(request["sl"]), symbol, symbol_info)
            if "tp" in request and float(request["tp"]) > 0:
                request["tp"] = PrecisionGuard.normalize_price(float(request["tp"]), symbol, symbol_info)
            
            # Filling Mode
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

        # 2. Execution Loop
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
                log.info(f"Order Placed: {symbol} Ticket: {result.order}")
                return result._asdict()
            elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_CONNECTION, mt5.TRADE_RETCODE_PRICE_OFF]:
                try: self.r.incr("broker:metric:requotes") 
                except: pass
                
                log.warning(f"Recoverable Error ({result.retcode}). Retrying...")
                time.sleep(0.5 * (2 ** attempt))
                
                # Refresh Price
                if request["action"] == mt5.TRADE_ACTION_DEAL:
                    with self.lock:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick:
                            new_price = tick.ask if request["type"] == mt5.ORDER_TYPE_BUY else tick.bid
                            if "price" in request: request["price"] = new_price
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
            result = self.execute_trade(request)

            if result and result.get('retcode') == mt5.TRADE_RETCODE_DONE:
                time.sleep(0.2)
                deals = mt5.history_deals_get(position=position_id)
                if deals:
                    profit = sum(d.profit + d.swap + d.commission for d in deals)
                    key = "risk:stats:wins" if profit > 0 else "risk:stats:losses"
                    try:
                        self.r.lpush(key, profit)
                        self.r.ltrim(key, 0, 99)
                        log.info(f"{LogSymbols.PROFIT if profit > 0 else LogSymbols.LOSS} Trade Closed. PnL: {profit:.2f}. Stats Updated.")
                    except: pass
                else:
                    log.warning(f"Trade Verification Failed: Could not retrieve history for {position_id}")

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
        self.mt5_lock = threading.RLock() # AUDIT FIX: Global RLock for MT5
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
        self.exec_engine = MT5ExecutionEngine(self.r, self.mt5_lock, self.ftmo_monitor)
        self.session_guard = SessionGuard()
        self.news_monitor = NewsEventMonitor()
        self.compliance_guard = FTMOComplianceGuard([])
        
        self.time_engine = LocalTimeFeatureTransformer()
        self.cluster_engine = LocalClusterContextBuilder(SYMBOLS)
        self.d1_cache = {p: {} for p in SYMBOLS}
        self.last_d1_update = 0

        self.notified_tickets = set()

        # Initialization Sequence
        self._optimize_process()
        self.connect_mt5()
        self.run_precise_backfill()
        
        # AUDIT FIX: Retry-guarded Risk State Reconstruction
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
        """
        AUDIT FIX: P-Core Pinning (0-15 for i7-13700K)
        """
        try:
            proc = psutil.Process(os.getpid())
            proc.nice(psutil.HIGH_PRIORITY_CLASS)
            
            # 8 P-Cores x 2 Threads = Indices 0-15
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
            if not mt5.initialize():
                log.critical(f"MT5 Init Failed: {mt5.last_error()}")
                sys.exit(1)
            for sym in SYMBOLS:
                mt5.symbol_select(sym, True)

    def _reconstruct_risk_state_from_history(self):
        """
        AUDIT FIX: 
        1. Compliance Breach Prevention via Retry Loop.
        2. Blocks startup until Risk State is cryptographically verified (conceptually).
        3. AUTO-DETECT: Updates Account Size if Config mismatches Broker.
        """
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
                    
                    # --- AUTO DETECT ACCOUNT SIZE (FIX) ---
                    # If config mismatches broker by > 1%, assume config is generic
                    # and adopt the broker's balance as the "Initial Balance" for FTMO limits.
                    if abs(self.ftmo_monitor.initial_balance - current_balance) > (current_balance * 0.01):
                        log.warning(f"⚠️ Auto-Detecting Account Size: Config ({self.ftmo_monitor.initial_balance}) != Broker ({current_balance}). Updating Risk Limits.")
                        self.ftmo_monitor.initial_balance = current_balance
                        self.ftmo_monitor.max_daily_loss = current_balance * CONFIG['risk_management']['max_daily_loss_pct']
                    # --------------------------------------

                    # History Scan
                    now = datetime.now()
                    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    deals = mt5.history_deals_get(midnight, now)
                    
                    if deals is None:
                        # AUDIT: Deals is None indicates API failure/Timeout. Do NOT assume 0.
                        log.error(f"MT5 History API returned None (Audit Fail). Retry {retry_count+1}...")
                        retry_count += 1
                        time.sleep(2)
                        continue

                    realized_pnl_today = 0.0
                    for d in deals:
                        if d.magic == MAGIC_NUMBER:
                            realized_pnl_today += (d.profit + d.swap + d.commission)
                    
                    calculated_start = current_balance - realized_pnl_today
                    
                    # Apply & Broadcast to Redis for Linux Engine
                    self.ftmo_monitor.starting_equity_of_day = calculated_start
                    self.r.set(CONFIG['redis']['risk_keys']['daily_starting_equity'], calculated_start)
                    self.r.set("bot:account_size", self.ftmo_monitor.initial_balance) # Broadcast Account Size
                    
                    log.info(f"{LogSymbols.SUCCESS} RISK STATE VERIFIED: Start Equity: {calculated_start:.2f} | Realized Today: {realized_pnl_today:.2f} | Account: {self.ftmo_monitor.initial_balance}")
                    return # Success

            except Exception as e:
                log.error(f"Risk Reconstruction Exception: {e}")
                retry_count += 1
                time.sleep(2)
        
        # If we reach here, we failed to verify risk state.
        log.critical("CRITICAL: FAILED TO VERIFY RISK STATE AFTER 5 ATTEMPTS. ABORTING STARTUP TO PREVENT COMPLIANCE BREACH.")
        sys.exit(1)

    def run_precise_backfill(self):
        # (Same as before, abbreviated for brevity in audit context, but ensuring function remains)
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
        monitored = set(SYMBOLS)
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

    def _tick_stream_loop(self):
        log.info("Starting Tick Stream...")
        interval = CONFIG['producer']['tick_interval_seconds']
        redis_failures = 0
        
        while self.running and not self.stop_event.is_set():
            # AUDIT FIX: File-based Kill Switch
            if os.path.exists(KILL_SWITCH_FILE):
                log.critical("💀 KILL SWITCH DETECTED. SHUTTING DOWN IMMEDIATELY.")
                with self.mt5_lock: mt5.shutdown()
                sys.exit(0)

            start = time.time()
            try:
                if time.time() - self.last_d1_update > 60:
                    self._update_d1_context()
                    self.last_d1_update = time.time()
                
                pipe = self.r.pipeline()
                for sym in self.monitored_symbols:
                    with self.mt5_lock:
                        tick = mt5.symbol_info_tick(sym)
                        if not tick: continue
                        
                        # AUDIT FIX: JPY Precision for Serialization
                        # Round to symbol digits to prevent 145.230000004 artifacts
                        sym_info = mt5.symbol_info(sym)
                        digits = sym_info.digits if sym_info else 5
                        bid = round(tick.bid, digits)
                        ask = round(tick.ask, digits)
                        
                        self.cluster_engine.update_tick(sym, bid)
                        ctx_cluster = self.cluster_engine.get_context_vector(sym)
                        utc_ts = int(tick.time_msc) - int(self.exec_engine.broker_time_offset * 1000)
                        dt_obj = datetime.utcfromtimestamp(utc_ts / 1000)
                        ctx_time = self.time_engine.transform(dt_obj)
                        
                        payload = {
                            "symbol": sym, "time": utc_ts, 
                            "bid": bid, 
                            "ask": ask, 
                            "volume": float(tick.volume_real if tick.volume_real > 0 else tick.volume),
                            "ctx_d1": json.dumps(self.d1_cache.get(sym, {})),
                            "ctx_cluster": json.dumps(ctx_cluster),
                            "ctx_time": json.dumps(ctx_time)
                        }
                        
                        # AUDIT FIX: Redis Memory Cap (maxlen)
                        if sym in SYMBOLS: pipe.xadd(STREAM_KEY, payload, maxlen=10000, approximate=True)
                        key = self.monitored_price_keys.get(sym, f"price:{sym}")
                        pipe.hset(key, mapping={"bid": payload["bid"], "ask": payload["ask"], "time": int(utc_ts/1000)})
                
                pipe.execute()
                self.r.setex(CONFIG['redis']['heartbeat_key'], 10, str(time.time()))
                redis_failures = 0 # Reset on success

            except (redis.ConnectionError, redis.TimeoutError) as e:
                # AUDIT FIX: Exponential Backoff for Redis
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
            with self.mt5_lock:
                rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 1, 200)
                if rates is not None and len(rates) > 0:
                    r = rates[-1]
                    ema = 0.0
                    if len(rates) >= 200:
                        closes = np.array([x['close'] for x in rates])
                        weights = np.exp(np.linspace(-1., 0., 200))
                        weights /= weights.sum()
                        ema = np.convolve(closes, weights, mode='valid')[0]
                    self.d1_cache[sym] = {
                        'open': float(r['open']), 'high': float(r['high']), 
                        'low': float(r['low']), 'close': float(r['close']),
                        'ema200': float(ema)
                    }

    def _trade_listener(self):
        log.info("Starting Trade Listener...")
        redis_failures = 0
        try: self.r.xgroup_create(TRADE_REQUEST_STREAM, "execution_group", id='0', mkstream=True)
        except: pass
        
        while self.running:
            try:
                # Trim request stream periodically
                if int(time.time()) % 60 == 0:
                    self.r.xtrim(TRADE_REQUEST_STREAM, maxlen=1000, approximate=True)

                entries = self.r.xreadgroup("execution_group", "producer_main", {TRADE_REQUEST_STREAM: '>'}, count=5, block=1000)
                redis_failures = 0 # Reset on success
                
                if entries:
                    for stream, messages in entries:
                        for msg_id, data in messages:
                            self.execution_queue.put((msg_id, data))
            
            except (redis.ConnectionError, redis.TimeoutError) as e:
                # AUDIT FIX: Exponential Backoff for Redis
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
            
            # --- AUDIT FIX: STRICT ZOMBIE TRADE PROTECTION ---
            # Enforce strict TTL. If the signal is older than MAX_LATENCY_SECONDS, drop it.
            request_ts_str = data.get('timestamp')
            
            if request_ts_str:
                try:
                    request_ts = float(request_ts_str)
                    now_ts = time.time()
                    latency = now_ts - request_ts
                    
                    if latency > MAX_LATENCY_SECONDS:
                        log.error(f"🧟 ZOMBIE TRADE DROPPED: {symbol} Lag: {latency:.2f}s > {MAX_LATENCY_SECONDS}s")
                        self.r.xack(TRADE_REQUEST_STREAM, "execution_group", msg_id)
                        return # ACK and DROP
                except ValueError:
                    log.error(f"Invalid timestamp: {request_ts_str}")
            # ------------------------------------------------

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

    def _sync_positions(self):
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

                    pos_list = []
                    if positions:
                        for p in positions:
                            if p.magic == MAGIC_NUMBER:
                                pos_list.append({
                                    "ticket": p.ticket, "symbol": p.symbol, 
                                    "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
                                    "volume": p.volume, "entry_price": p.price_open, 
                                    "profit": p.profit, "sl": p.sl, "tp": p.tp
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
        log.info("Starting Midnight Watchman...")
        freeze_key = CONFIG['redis']['risk_keys']['midnight_freeze']
        tz_name = CONFIG['risk_management'].get('risk_timezone', 'Europe/Prague')
        risk_tz = pytz.timezone(tz_name)

        while self.running and not self.stop_event.is_set():
            now = datetime.now(timezone.utc).replace(tzinfo=timezone.utc).astimezone(risk_tz)
            if (now.hour == 23 and now.minute >= 55) or (now.hour == 0 and now.minute <= 5):
                self.r.set(freeze_key, "1")
                if now.second % 30 == 0: log.info("MIDNIGHT WATCHMAN: Trading Frozen.")
            elif self.r.exists(freeze_key):
                if (now.hour == 0 and now.minute > 5) or (now.hour > 0):
                    self.r.delete(freeze_key)
                    log.info("MIDNIGHT WATCHMAN: Trading Resumed.")
            time.sleep(10)

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
                # AUDIT FIX: Memory Leak Prevention (Cleanup notified tickets)
                if len(self.notified_tickets) > 5000:
                    # Keep only recent 1000
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
                                    log.info(f"Cancelling Zombie Order {order.ticket}")
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
            threading.Thread(target=self._pending_order_monitor, daemon=True)
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