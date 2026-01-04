#!/usr/bin/env python3
"""
Backend Backtest Script for DOGE - WITH PHASE 20 RISK MANAGEMENT
Uses the same signal generation and risk management as the live system.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL = "DOGEUSDT"
INITIAL_BALANCE = 10000.0
LEVERAGE = 100
RISK_PER_TRADE = 0.02  # 2%
SL_ATR = 2.0
TP_ATR = 3.0
TRAIL_ACTIVATION_ATR = 1.5
TRAIL_DISTANCE_ATR = 1.0

# Phase 20: Risk Management
MAX_POSITION_AGE_HOURS = 24
EMERGENCY_SL_PCT = 10.0  # %10 max loss
DAILY_DRAWDOWN_LIMIT = 5.0  # %5

# Backtest period
DAYS_BACK = 30
INTERVAL = "15m"

# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_historical_klines(symbol: str, interval: str, days: int):
    """Fetch historical klines from Binance."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    all_klines = []
    current_start = start_time
    
    print(f"📊 Fetching {days} days of {symbol} {interval} data...")
    
    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000
        }
        
        response = requests.get(url, params=params)
        klines = response.json()
        
        if not klines:
            break
            
        all_klines.extend(klines)
        current_start = klines[-1][6] + 1
        
    print(f"✅ Fetched {len(all_klines)} candles")
    
    df = pd.DataFrame(all_klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ])
    
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    
    return df

# ============================================================================
# INDICATORS
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_hurst(series: pd.Series, lags: list = None) -> float:
    if lags is None:
        lags = range(2, min(20, len(series) // 2))
    
    try:
        tau = []
        for lag in lags:
            pp = series.diff(lag).dropna()
            if len(pp) > 0:
                tau.append(np.std(pp))
        
        if len(tau) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        return reg[0]
    except:
        return 0.5

def calculate_zscore(series: pd.Series, period: int = 20) -> pd.Series:
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    zscore = (series - mean) / std
    return zscore

# ============================================================================
# SIGNAL GENERATION
# ============================================================================
def generate_signal(row: pd.Series, atr: float, hurst: float, zscore: float) -> dict:
    if pd.isna(atr) or atr == 0:
        return None
    
    signal = None
    
    # Mean Reversion Logic (Hurst < 0.45)
    if hurst < 0.45:
        if zscore < -2.0:
            signal = {
                "action": "LONG",
                "entry": row['close'],
                "sl": row['close'] - (SL_ATR * atr),
                "tp": row['close'] + (TP_ATR * atr),
                "reason": f"MR Long: Z={zscore:.2f}, H={hurst:.2f}"
            }
        elif zscore > 2.0:
            signal = {
                "action": "SHORT",
                "entry": row['close'],
                "sl": row['close'] + (SL_ATR * atr),
                "tp": row['close'] - (TP_ATR * atr),
                "reason": f"MR Short: Z={zscore:.2f}, H={hurst:.2f}"
            }
    
    # Trend Following Logic (Hurst > 0.55)
    elif hurst > 0.55:
        if zscore > 1.5:
            signal = {
                "action": "LONG",
                "entry": row['close'],
                "sl": row['close'] - (SL_ATR * atr),
                "tp": row['close'] + (TP_ATR * atr),
                "reason": f"TF Long: Z={zscore:.2f}, H={hurst:.2f}"
            }
        elif zscore < -1.5:
            signal = {
                "action": "SHORT",
                "entry": row['close'],
                "sl": row['close'] + (SL_ATR * atr),
                "tp": row['close'] - (TP_ATR * atr),
                "reason": f"TF Short: Z={zscore:.2f}, H={hurst:.2f}"
            }
    
    return signal

# ============================================================================
# PHASE 20: BACKTEST ENGINE WITH RISK MANAGEMENT
# ============================================================================
class BacktestEngineV2:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = 0
        self.trading_paused = False
        
    def get_dynamic_trail_distance(self, atr: float, spread_pct: float = 0.05) -> float:
        """Phase 20: Spread-aware trailing."""
        if spread_pct < 0.05:
            return atr * 0.5
        elif spread_pct < 0.15:
            return atr * 1.0
        else:
            return atr * (1.0 + spread_pct)
    
    def update_progressive_sl(self, pos: dict, current_price: float, atr: float) -> bool:
        """Phase 20: Progressive SL - lock profits."""
        entry = pos["entry_price"]
        
        if pos["side"] == "LONG":
            profit_atr = (current_price - entry) / atr if atr > 0 else 0
            
            if profit_atr >= 4:
                new_sl = entry + (2.5 * atr)
            elif profit_atr >= 3:
                new_sl = entry + (1.5 * atr)
            elif profit_atr >= 2:
                new_sl = entry + (0.5 * atr)
            elif profit_atr >= 1:
                new_sl = entry  # Breakeven
            else:
                return False
                
            if new_sl > pos["sl"]:
                pos["sl"] = new_sl
                return True
                
        elif pos["side"] == "SHORT":
            profit_atr = (entry - current_price) / atr if atr > 0 else 0
            
            if profit_atr >= 4:
                new_sl = entry - (2.5 * atr)
            elif profit_atr >= 3:
                new_sl = entry - (1.5 * atr)
            elif profit_atr >= 2:
                new_sl = entry - (0.5 * atr)
            elif profit_atr >= 1:
                new_sl = entry
            else:
                return False
                
            if new_sl < pos["sl"]:
                pos["sl"] = new_sl
                return True
        
        return False
    
    def check_emergency_sl(self, pos: dict, current_price: float) -> bool:
        """Phase 20: Emergency SL at max loss."""
        entry = pos["entry_price"]
        
        if pos["side"] == "LONG":
            loss_pct = ((entry - current_price) / entry) * 100
        else:
            loss_pct = ((current_price - entry) / entry) * 100
            
        return loss_pct >= EMERGENCY_SL_PCT
    
    def open_position(self, signal: dict, timestamp, current_price: float, atr: float):
        if self.position is not None or self.trading_paused:
            return
            
        risk_amount = self.balance * RISK_PER_TRADE
        position_size_usd = risk_amount * LEVERAGE
        position_size = position_size_usd / current_price
        
        self.position = {
            "side": signal["action"],
            "entry_price": current_price,
            "size": position_size,
            "size_usd": position_size_usd,
            "sl": signal["sl"],
            "tp": signal["tp"],
            "trail_sl": signal["sl"],
            "trail_active": False,
            "open_time": timestamp,
            "reason": signal["reason"],
            "atr": atr
        }
        
    def update_position(self, current_price: float, high: float, low: float, timestamp, atr: float):
        if self.position is None:
            return
            
        pos = self.position
        
        # Phase 20: Emergency SL check first
        if self.check_emergency_sl(pos, current_price):
            self.close_position(current_price, timestamp, "EMERGENCY_SL")
            return
        
        # Phase 20: Progressive SL
        self.update_progressive_sl(pos, current_price, atr)
        
        # Dynamic trailing distance
        trail_dist = self.get_dynamic_trail_distance(atr)
        
        if pos["side"] == "LONG":
            # Trailing activation
            activation_price = pos["entry_price"] + (TRAIL_ACTIVATION_ATR * atr)
            if current_price >= activation_price:
                pos["trail_active"] = True
                
            if pos["trail_active"]:
                new_sl = current_price - trail_dist
                if new_sl > pos["trail_sl"]:
                    pos["trail_sl"] = new_sl
                    pos["sl"] = new_sl
            
            # Check exits
            if low <= pos["sl"]:
                self.close_position(pos["sl"], timestamp, "SL")
            elif high >= pos["tp"]:
                self.close_position(pos["tp"], timestamp, "TP")
                
        else:  # SHORT
            activation_price = pos["entry_price"] - (TRAIL_ACTIVATION_ATR * atr)
            if current_price <= activation_price:
                pos["trail_active"] = True
                
            if pos["trail_active"]:
                new_sl = current_price + trail_dist
                if new_sl < pos["trail_sl"]:
                    pos["trail_sl"] = new_sl
                    pos["sl"] = new_sl
            
            if high >= pos["sl"]:
                self.close_position(pos["sl"], timestamp, "SL")
            elif low <= pos["tp"]:
                self.close_position(pos["tp"], timestamp, "TP")
                
    def close_position(self, exit_price: float, timestamp, reason: str):
        if self.position is None:
            return
            
        pos = self.position
        
        if pos["side"] == "LONG":
            pnl = (exit_price - pos["entry_price"]) * pos["size"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["size"]
            
        self.balance += pnl
        self.daily_pnl += pnl
        
        # Phase 20: Daily drawdown check
        daily_pnl_pct = (self.daily_pnl / self.initial_balance) * 100
        if daily_pnl_pct < -DAILY_DRAWDOWN_LIMIT:
            self.trading_paused = True
        
        trade = {
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": (pnl / pos["size_usd"]) * 100,
            "open_time": pos["open_time"],
            "close_time": timestamp,
            "reason": reason,
            "signal_reason": pos["reason"],
            "trail_active": pos["trail_active"]
        }
        self.trades.append(trade)
        self.position = None
        
    def record_equity(self, timestamp, current_price: float):
        unrealized = 0
        if self.position:
            if self.position["side"] == "LONG":
                unrealized = (current_price - self.position["entry_price"]) * self.position["size"]
            else:
                unrealized = (self.position["entry_price"] - current_price) * self.position["size"]
                
        self.equity_curve.append({
            "time": timestamp,
            "equity": self.balance + unrealized
        })
        
    def get_stats(self) -> dict:
        if not self.trades:
            return {"error": "No trades"}
            
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        
        total_pnl = sum(t["pnl"] for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100
        
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Trailing analysis
        trail_trades = [t for t in self.trades if t["trail_active"]]
        trail_pnl = sum(t["pnl"] for t in trail_trades)
        
        # Exit reason breakdown
        exit_reasons = {}
        for t in self.trades:
            r = t["reason"]
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t["pnl"]
        
        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for eq in self.equity_curve:
            if eq["equity"] > peak:
                peak = eq["equity"]
            dd = (peak - eq["equity"]) / peak * 100
            if dd > max_dd:
                max_dd = dd
                
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(self.balance, 2),
            "return_percent": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 2),
            "trailing_trades": len(trail_trades),
            "trailing_pnl": round(trail_pnl, 2),
            "exit_reasons": exit_reasons
        }

# ============================================================================
# MAIN BACKTEST
# ============================================================================
def run_backtest():
    print("=" * 60)
    print(f"🚀 BACKTEST: {SYMBOL} (Phase 20 Risk Management)")
    print(f"📅 Period: {DAYS_BACK} days")
    print(f"⚙️  Settings: {LEVERAGE}x | SL:{SL_ATR} ATR | TP:{TP_ATR} ATR")
    print(f"🛡️  Risk: Emergency SL:{EMERGENCY_SL_PCT}% | Daily DD:{DAILY_DRAWDOWN_LIMIT}%")
    print("=" * 60)
    
    df = fetch_historical_klines(SYMBOL, INTERVAL, DAYS_BACK)
    
    if len(df) < 50:
        print("❌ Not enough data")
        return
    
    print("📊 Calculating indicators...")
    df['atr'] = calculate_atr(df, 14)
    df['zscore'] = calculate_zscore(df['close'], 20)
    
    engine = BacktestEngineV2(INITIAL_BALANCE)
    
    print("🔄 Running backtest with Phase 20 risk management...")
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        current_price = row['close']
        atr = row['atr']
        
        if i >= 100:
            hurst = calculate_hurst(df['close'].iloc[i-100:i])
        else:
            hurst = 0.5
            
        zscore = row['zscore']
        
        # Update existing position with Phase 20 risk management
        engine.update_position(current_price, row['high'], row['low'], row['timestamp'], atr)
        
        # Generate signal if no position
        if engine.position is None and not engine.trading_paused:
            signal = generate_signal(row, atr, hurst, zscore)
            if signal:
                engine.open_position(signal, row['timestamp'], current_price, atr)
        
        if i % 10 == 0:
            engine.record_equity(row['timestamp'], current_price)
    
    if engine.position:
        engine.close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "END")
    
    stats = engine.get_stats()
    
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS (Phase 20)")
    print("=" * 60)
    print(f"Total Trades:    {stats['total_trades']}")
    print(f"Winning:         {stats['winning_trades']}")
    print(f"Losing:          {stats['losing_trades']}")
    print(f"Win Rate:        {stats['win_rate']}%")
    print("-" * 40)
    print(f"Total PnL:       ${stats['total_pnl']}")
    print(f"Final Balance:   ${stats['final_balance']}")
    print(f"Return:          {stats['return_percent']}%")
    print("-" * 40)
    print(f"Avg Win:         ${stats['avg_win']}")
    print(f"Avg Loss:        ${stats['avg_loss']}")
    print(f"Profit Factor:   {stats['profit_factor']}")
    print(f"Max Drawdown:    {stats['max_drawdown']}%")
    print("-" * 40)
    print(f"Trailing Trades: {stats['trailing_trades']}")
    print(f"Trailing PnL:    ${stats['trailing_pnl']}")
    print("=" * 60)
    
    print("\n📊 Exit Reason Breakdown:")
    for reason, data in stats['exit_reasons'].items():
        pnl_emoji = "✅" if data['pnl'] > 0 else "❌"
        print(f"  {reason}: {data['count']} trades | {pnl_emoji} ${data['pnl']:.2f}")
    
    print("\n📋 Last 10 Trades:")
    for trade in engine.trades[-10:]:
        emoji = "✅" if trade["pnl"] > 0 else "❌"
        trail = "🔄" if trade["trail_active"] else ""
        print(f"{emoji}{trail} {trade['side']} @ ${trade['entry_price']:.6f} → ${trade['exit_price']:.6f} | PnL: ${trade['pnl']:.2f} ({trade['reason']})")
    
    return stats, engine.trades

if __name__ == "__main__":
    run_backtest()
