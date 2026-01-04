#!/usr/bin/env python3
"""
Backend Backtest Script for DOGE
Uses the same signal generation logic as the live system.
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
        current_start = klines[-1][6] + 1  # Close time + 1ms
        
    print(f"✅ Fetched {len(all_klines)} candles")
    
    # Convert to DataFrame
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
# INDICATORS (Same as live system)
# ============================================================================
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
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
    """Calculate Hurst Exponent."""
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
    """Calculate Z-Score."""
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    zscore = (series - mean) / std
    return zscore

# ============================================================================
# SIGNAL GENERATION (Same logic as live system)
# ============================================================================
def generate_signal(row: pd.Series, atr: float, hurst: float, zscore: float) -> dict:
    """Generate trading signal based on current conditions."""
    
    # Skip if conditions not met
    if pd.isna(atr) or atr == 0:
        return None
    
    signal = None
    
    # Mean Reversion Logic (when Hurst < 0.5)
    if hurst < 0.45:
        if zscore < -2.0:  # Oversold
            signal = {
                "action": "LONG",
                "entry": row['close'],
                "sl": row['close'] - (SL_ATR * atr),
                "tp": row['close'] + (TP_ATR * atr),
                "reason": f"MR Long: Z={zscore:.2f}, H={hurst:.2f}"
            }
        elif zscore > 2.0:  # Overbought
            signal = {
                "action": "SHORT",
                "entry": row['close'],
                "sl": row['close'] + (SL_ATR * atr),
                "tp": row['close'] - (TP_ATR * atr),
                "reason": f"MR Short: Z={zscore:.2f}, H={hurst:.2f}"
            }
    
    # Trend Following Logic (when Hurst > 0.55)
    elif hurst > 0.55:
        if zscore > 1.5:  # Strong uptrend
            signal = {
                "action": "LONG",
                "entry": row['close'],
                "sl": row['close'] - (SL_ATR * atr),
                "tp": row['close'] + (TP_ATR * atr),
                "reason": f"TF Long: Z={zscore:.2f}, H={hurst:.2f}"
            }
        elif zscore < -1.5:  # Strong downtrend
            signal = {
                "action": "SHORT",
                "entry": row['close'],
                "sl": row['close'] + (SL_ATR * atr),
                "tp": row['close'] - (TP_ATR * atr),
                "reason": f"TF Short: Z={zscore:.2f}, H={hurst:.2f}"
            }
    
    return signal

# ============================================================================
# BACKTEST ENGINE
# ============================================================================
class BacktestEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    def open_position(self, signal: dict, timestamp, current_price: float):
        if self.position is not None:
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
            "reason": signal["reason"]
        }
        
    def update_position(self, current_price: float, high: float, low: float, timestamp):
        if self.position is None:
            return
            
        pos = self.position
        
        # Calculate unrealized PnL
        if pos["side"] == "LONG":
            pnl = (current_price - pos["entry_price"]) * pos["size"]
            
            # Check trailing activation
            atr_move = (current_price - pos["entry_price"]) / (pos["entry_price"] * 0.01)  # Rough ATR estimate
            if atr_move >= TRAIL_ACTIVATION_ATR and not pos["trail_active"]:
                pos["trail_active"] = True
                
            # Update trailing stop
            if pos["trail_active"]:
                new_sl = current_price - (TRAIL_DISTANCE_ATR * (pos["entry_price"] * 0.01))
                if new_sl > pos["trail_sl"]:
                    pos["trail_sl"] = new_sl
                    pos["sl"] = new_sl
            
            # Check exits
            if low <= pos["sl"]:
                self.close_position(pos["sl"], timestamp, "SL")
            elif high >= pos["tp"]:
                self.close_position(pos["tp"], timestamp, "TP")
                
        else:  # SHORT
            pnl = (pos["entry_price"] - current_price) * pos["size"]
            
            # Trailing for shorts
            atr_move = (pos["entry_price"] - current_price) / (pos["entry_price"] * 0.01)
            if atr_move >= TRAIL_ACTIVATION_ATR and not pos["trail_active"]:
                pos["trail_active"] = True
                
            if pos["trail_active"]:
                new_sl = current_price + (TRAIL_DISTANCE_ATR * (pos["entry_price"] * 0.01))
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
        
        trade = {
            "side": pos["side"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_percent": (pnl / pos["size_usd"]) * 100,
            "open_time": pos["open_time"],
            "close_time": timestamp,
            "reason": reason,
            "signal_reason": pos["reason"]
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
            "max_drawdown": round(max_dd, 2)
        }

# ============================================================================
# MAIN BACKTEST
# ============================================================================
def run_backtest():
    print("=" * 60)
    print(f"🚀 BACKTEST: {SYMBOL}")
    print(f"📅 Period: {DAYS_BACK} days")
    print(f"⚙️  Settings: {LEVERAGE}x | SL:{SL_ATR} ATR | TP:{TP_ATR} ATR")
    print("=" * 60)
    
    # Fetch data
    df = fetch_historical_klines(SYMBOL, INTERVAL, DAYS_BACK)
    
    if len(df) < 50:
        print("❌ Not enough data")
        return
    
    # Calculate indicators
    print("📊 Calculating indicators...")
    df['atr'] = calculate_atr(df, 14)
    df['zscore'] = calculate_zscore(df['close'], 20)
    
    # Initialize backtest engine
    engine = BacktestEngine(INITIAL_BALANCE)
    
    # Run backtest
    print("🔄 Running backtest...")
    
    for i in range(50, len(df)):
        row = df.iloc[i]
        current_price = row['close']
        atr = row['atr']
        
        # Calculate Hurst on rolling window
        if i >= 100:
            hurst = calculate_hurst(df['close'].iloc[i-100:i])
        else:
            hurst = 0.5
            
        zscore = row['zscore']
        
        # Update existing position
        engine.update_position(current_price, row['high'], row['low'], row['timestamp'])
        
        # Generate signal if no position
        if engine.position is None:
            signal = generate_signal(row, atr, hurst, zscore)
            if signal:
                engine.open_position(signal, row['timestamp'], current_price)
        
        # Record equity every 10 candles
        if i % 10 == 0:
            engine.record_equity(row['timestamp'], current_price)
    
    # Close any open position at the end
    if engine.position:
        engine.close_position(df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "END")
    
    # Get and print results
    stats = engine.get_stats()
    
    print("\n" + "=" * 60)
    print("📈 BACKTEST RESULTS")
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
    print("=" * 60)
    
    # Print last 10 trades
    print("\n📋 Last 10 Trades:")
    for trade in engine.trades[-10:]:
        emoji = "✅" if trade["pnl"] > 0 else "❌"
        print(f"{emoji} {trade['side']} | Entry: ${trade['entry_price']:.6f} → Exit: ${trade['exit_price']:.6f} | PnL: ${trade['pnl']:.2f} ({trade['reason']})")
    
    return stats, engine.trades

if __name__ == "__main__":
    run_backtest()
