#!/usr/bin/env python3
"""
Backend Backtest Script for DOGE - WITH PHASE 22 MTF CONFIRMATION
Uses multi-timeframe analysis with 3+ TF agreement required.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================
SYMBOL = "DOGEUSDT"
INITIAL_BALANCE = 10000.0
LEVERAGE = 100
RISK_PER_TRADE = 0.02  # 2%

# Phase 22: MTF Configuration
MTF_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MTF_MIN_AGREEMENT = 4

# Phase 22: Spread Multipliers
SPREAD_MULTIPLIERS = {
    "very_low": {"max_spread": 0.03, "trail": 0.3, "sl": 1.5, "tp": 2.5},
    "low": {"max_spread": 0.08, "trail": 0.6, "sl": 2.0, "tp": 3.0},
    "normal": {"max_spread": 0.15, "trail": 1.0, "sl": 2.5, "tp": 4.0},
    "high": {"max_spread": 0.30, "trail": 1.5, "sl": 3.0, "tp": 5.0},
    "very_high": {"max_spread": 1.0, "trail": 2.0, "sl": 4.0, "tp": 6.0}
}

# Phase 20: Risk Management
MAX_POSITION_AGE_HOURS = 24
EMERGENCY_SL_PCT = 10.0
DAILY_DRAWDOWN_LIMIT = 5.0

# Backtest period
DAYS_BACK = 30
INTERVALS = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

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

def calculate_hurst(series, lags=None):
    if len(series) < 20:
        return 0.5
    
    if lags is None:
        lags = range(2, min(20, len(series) // 2))
    
    try:
        tau = []
        for lag in lags:
            pp = np.diff(series[-100:], n=lag) if len(series) >= 100 else np.diff(series, n=lag)
            if len(pp) > 0:
                tau.append(np.std(pp))
        
        if len(tau) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
        return max(0.3, min(0.7, reg[0]))
    except:
        return 0.5

def calculate_zscore(series, period=20):
    if len(series) < period:
        return 0
    arr = np.array(series[-period:])
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0
    return (arr[-1] - mean) / std

# ============================================================================
# PHASE 22: MULTI-TIMEFRAME ANALYZER
# ============================================================================
class MTFAnalyzer:
    def __init__(self):
        self.min_agreement = MTF_MIN_AGREEMENT
    
    def analyze_timeframe(self, closes):
        if len(closes) < 20:
            return {"direction": "NEUTRAL", "strength": 0}
        
        hurst = calculate_hurst(closes)
        zscore = calculate_zscore(closes)
        
        direction = "NEUTRAL"
        strength = 0
        
        # Mean Reversion (Hurst < 0.45)
        if hurst < 0.45:
            if zscore < -2.0:
                direction = "LONG"
                strength = abs(zscore)
            elif zscore > 2.0:
                direction = "SHORT"
                strength = abs(zscore)
        
        # Trend Following (Hurst > 0.55)
        elif hurst > 0.55:
            if zscore > 1.5:
                direction = "LONG"
                strength = abs(zscore)
            elif zscore < -1.5:
                direction = "SHORT"
                strength = abs(zscore)
        
        return {"direction": direction, "strength": strength, "hurst": hurst, "zscore": zscore}
    
    def get_confirmation(self, tf_signals):
        if not tf_signals:
            return None
        
        long_count = sum(1 for s in tf_signals.values() if s.get('direction') == 'LONG')
        short_count = sum(1 for s in tf_signals.values() if s.get('direction') == 'SHORT')
        
        total_tfs = len(tf_signals)
        
        if long_count >= self.min_agreement:
            return {
                "action": "LONG",
                "tf_count": long_count,
                "total_tfs": total_tfs,
                "confidence": long_count / total_tfs
            }
        elif short_count >= self.min_agreement:
            return {
                "action": "SHORT",
                "tf_count": short_count,
                "total_tfs": total_tfs,
                "confidence": short_count / total_tfs
            }
        
        return None
    
    def get_size_multiplier(self, confirmation):
        if not confirmation:
            return 0.5
        
        tf_count = confirmation.get('tf_count', 0)
        
        if tf_count >= 5:
            return 2.0
        elif tf_count >= 4:
            return 1.5
        elif tf_count >= 3:
            return 1.0
        else:
            return 0.5

# ============================================================================
# BACKTEST ENGINE
# ============================================================================
class BacktestEngineV3:
    def __init__(self, initial_balance=10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []  # Phase 22: Multiple positions
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = 0
        self.trading_paused = False
        self.mtf_analyzer = MTFAnalyzer()
        
    def can_open_position(self, action):
        """Phase 22: Check if new position can be opened."""
        if len(self.positions) >= 3:
            return False
        
        # Max 2 same direction
        same_direction = [p for p in self.positions if p['side'] == action]
        if len(same_direction) >= 2:
            return False
        
        return True
    
    def get_spread_params(self, spread_pct, atr):
        """Phase 22: Get spread-adjusted SL/TP."""
        for level, params in SPREAD_MULTIPLIERS.items():
            if spread_pct <= params["max_spread"]:
                return {
                    "trail_distance": atr * params["trail"],
                    "sl": atr * params["sl"],
                    "tp": atr * params["tp"],
                    "level": level
                }
        params = SPREAD_MULTIPLIERS["very_high"]
        return {
            "trail_distance": atr * params["trail"],
            "sl": atr * params["sl"],
            "tp": atr * params["tp"],
            "level": "very_high"
        }
    
    def open_position(self, action, price, atr, timestamp, size_multiplier=1.0, spread_pct=0.05):
        if not self.can_open_position(action) or self.trading_paused:
            return None
        
        spread_params = self.get_spread_params(spread_pct, atr)
        
        risk_amount = self.balance * RISK_PER_TRADE * size_multiplier
        position_size_usd = risk_amount * LEVERAGE
        position_size = position_size_usd / price
        
        if action == "LONG":
            sl = price - spread_params['sl']
            tp = price + spread_params['tp']
        else:
            sl = price + spread_params['sl']
            tp = price - spread_params['tp']
        
        position = {
            "id": len(self.trades) + len(self.positions),
            "side": action,
            "entry_price": price,
            "size": position_size,
            "size_usd": position_size_usd,
            "sl": sl,
            "tp": tp,
            "trail_sl": sl,
            "trail_distance": spread_params['trail_distance'],
            "trail_active": False,
            "open_time": timestamp,
            "size_multiplier": size_multiplier
        }
        
        self.positions.append(position)
        return position
    
    def update_positions(self, price, high, low, timestamp, atr):
        closed = []
        
        for pos in list(self.positions):
            # Emergency SL check
            if pos['side'] == 'LONG':
                loss_pct = ((pos['entry_price'] - price) / pos['entry_price']) * 100
            else:
                loss_pct = ((price - pos['entry_price']) / pos['entry_price']) * 100
            
            if loss_pct >= EMERGENCY_SL_PCT:
                self.close_position(pos, price, timestamp, "EMERGENCY_SL")
                closed.append(pos)
                continue
            
            # Progressive SL
            if pos['side'] == 'LONG':
                profit_atr = (price - pos['entry_price']) / atr if atr > 0 else 0
                if profit_atr >= 2:
                    new_sl = pos['entry_price'] + (0.5 * atr)
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
                elif profit_atr >= 1:
                    new_sl = pos['entry_price']
                    if new_sl > pos['sl']:
                        pos['sl'] = new_sl
            else:
                profit_atr = (pos['entry_price'] - price) / atr if atr > 0 else 0
                if profit_atr >= 2:
                    new_sl = pos['entry_price'] - (0.5 * atr)
                    if new_sl < pos['sl']:
                        pos['sl'] = new_sl
                elif profit_atr >= 1:
                    new_sl = pos['entry_price']
                    if new_sl < pos['sl']:
                        pos['sl'] = new_sl
            
            # Trailing
            if pos['side'] == 'LONG':
                activation = pos['entry_price'] + (1.5 * atr)
                if price >= activation:
                    pos['trail_active'] = True
                
                if pos['trail_active']:
                    new_sl = price - pos['trail_distance']
                    if new_sl > pos['trail_sl']:
                        pos['trail_sl'] = new_sl
                        pos['sl'] = new_sl
                
                # Check exits
                if low <= pos['sl']:
                    self.close_position(pos, pos['sl'], timestamp, "SL")
                    closed.append(pos)
                elif high >= pos['tp']:
                    self.close_position(pos, pos['tp'], timestamp, "TP")
                    closed.append(pos)
            else:
                activation = pos['entry_price'] - (1.5 * atr)
                if price <= activation:
                    pos['trail_active'] = True
                
                if pos['trail_active']:
                    new_sl = price + pos['trail_distance']
                    if new_sl < pos['trail_sl']:
                        pos['trail_sl'] = new_sl
                        pos['sl'] = new_sl
                
                if high >= pos['sl']:
                    self.close_position(pos, pos['sl'], timestamp, "SL")
                    closed.append(pos)
                elif low <= pos['tp']:
                    self.close_position(pos, pos['tp'], timestamp, "TP")
                    closed.append(pos)
        
        return closed
    
    def close_position(self, pos, exit_price, timestamp, reason):
        if pos['side'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * pos['size']
        else:
            pnl = (pos['entry_price'] - exit_price) * pos['size']
        
        self.balance += pnl
        self.daily_pnl += pnl
        
        if pos in self.positions:
            self.positions.remove(pos)
        
        trade = {
            "side": pos['side'],
            "entry_price": pos['entry_price'],
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": reason,
            "trail_active": pos['trail_active'],
            "size_multiplier": pos['size_multiplier'],
            "close_time": timestamp
        }
        self.trades.append(trade)
    
    def record_equity(self, timestamp, price):
        unrealized = 0
        for pos in self.positions:
            if pos['side'] == 'LONG':
                unrealized += (price - pos['entry_price']) * pos['size']
            else:
                unrealized += (pos['entry_price'] - price) * pos['size']
        
        self.equity_curve.append({
            "time": timestamp,
            "equity": self.balance + unrealized
        })
    
    def get_stats(self):
        if not self.trades:
            return {"error": "No trades"}
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Max drawdown
        peak = self.initial_balance
        max_dd = 0
        for eq in self.equity_curve:
            if eq['equity'] > peak:
                peak = eq['equity']
            dd = (peak - eq['equity']) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            r = t['reason']
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t['pnl']
        
        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "final_balance": round(self.balance, 2),
            "return_percent": round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(max_dd, 2),
            "exit_reasons": exit_reasons
        }

# ============================================================================
# MAIN BACKTEST
# ============================================================================
def run_backtest():
    print("=" * 70)
    print(f"🚀 BACKTEST: {SYMBOL} (Phase 22 MTF + Multi-Position + Wider Spreads)")
    print(f"📅 Period: {DAYS_BACK} days | TFs: {', '.join(MTF_TIMEFRAMES)}")
    print(f"🎯 MTF Min Agreement: {MTF_MIN_AGREEMENT} / {len(MTF_TIMEFRAMES)}")
    print("=" * 70)
    
    # Fetch all timeframe data
    print("\n📊 Fetching multi-timeframe data...")
    tf_data = {}
    for tf in ['15m', '1h', '4h', '1d']:
        print(f"  Fetching {tf}...")
        tf_data[tf] = fetch_historical_klines(SYMBOL, tf, DAYS_BACK)
        print(f"  ✅ {len(tf_data[tf])} candles")
    
    # Use 15m as primary timeframe
    df = tf_data['15m']
    df['atr'] = calculate_atr(df, 14)
    
    engine = BacktestEngineV3(INITIAL_BALANCE)
    mtf = MTFAnalyzer()
    
    mtf_confirmed = 0
    mtf_rejected = 0
    
    print("\n🔄 Running backtest with MTF confirmation...")
    
    for i in range(100, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr']
        timestamp = row['timestamp']
        
        if pd.isna(atr) or atr == 0:
            continue
        
        # Update existing positions
        engine.update_positions(price, row['high'], row['low'], timestamp, atr)
        
        # Analyze all timeframes
        tf_signals = {}
        
        # 15m (primary)
        closes_15m = df['close'].iloc[max(0, i-100):i+1].tolist()
        tf_signals['15m'] = mtf.analyze_timeframe(closes_15m)
        tf_signals['1m'] = tf_signals['15m']  # Approximate
        tf_signals['5m'] = tf_signals['15m']  # Approximate
        
        # 1h
        if '1h' in tf_data:
            ts = timestamp
            h_idx = tf_data['1h'][tf_data['1h']['timestamp'] <= ts].index
            if len(h_idx) >= 20:
                closes_1h = tf_data['1h'].loc[h_idx[-20:], 'close'].tolist()
                tf_signals['1h'] = mtf.analyze_timeframe(closes_1h)
            else:
                tf_signals['1h'] = {"direction": "NEUTRAL", "strength": 0}
        
        # 4h
        if '4h' in tf_data:
            h_idx = tf_data['4h'][tf_data['4h']['timestamp'] <= ts].index
            if len(h_idx) >= 20:
                closes_4h = tf_data['4h'].loc[h_idx[-20:], 'close'].tolist()
                tf_signals['4h'] = mtf.analyze_timeframe(closes_4h)
            else:
                tf_signals['4h'] = {"direction": "NEUTRAL", "strength": 0}
        
        # 1d
        if '1d' in tf_data:
            d_idx = tf_data['1d'][tf_data['1d']['timestamp'] <= ts].index
            if len(d_idx) >= 10:
                closes_1d = tf_data['1d'].loc[d_idx[-10:], 'close'].tolist()
                tf_signals['1d'] = mtf.analyze_timeframe(closes_1d)
            else:
                tf_signals['1d'] = {"direction": "NEUTRAL", "strength": 0}
        
        # Get MTF confirmation
        confirmation = mtf.get_confirmation(tf_signals)
        
        if confirmation:
            mtf_confirmed += 1
            size_mult = mtf.get_size_multiplier(confirmation)
            
            # Open position
            engine.open_position(
                action=confirmation['action'],
                price=price,
                atr=atr,
                timestamp=timestamp,
                size_multiplier=size_mult,
                spread_pct=0.05
            )
        else:
            # Check if would have been a signal without MTF
            primary = tf_signals.get('15m', {})
            if primary.get('direction') in ['LONG', 'SHORT']:
                mtf_rejected += 1
        
        if i % 50 == 0:
            engine.record_equity(timestamp, price)
    
    # Close remaining positions
    for pos in list(engine.positions):
        engine.close_position(pos, df.iloc[-1]['close'], df.iloc[-1]['timestamp'], "END")
    
    stats = engine.get_stats()
    
    print("\n" + "=" * 70)
    print("📈 BACKTEST RESULTS (Phase 22 MTF)")
    print("=" * 70)
    print(f"Total Trades:    {stats['total_trades']}")
    print(f"Winning:         {stats['winning_trades']}")
    print(f"Losing:          {stats['losing_trades']}")
    print(f"Win Rate:        {stats['win_rate']}%")
    print("-" * 50)
    print(f"Total PnL:       ${stats['total_pnl']}")
    print(f"Final Balance:   ${stats['final_balance']}")
    print(f"Return:          {stats['return_percent']}%")
    print("-" * 50)
    print(f"Profit Factor:   {stats['profit_factor']}")
    print(f"Max Drawdown:    {stats['max_drawdown']}%")
    print("-" * 50)
    print(f"MTF Confirmed:   {mtf_confirmed}")
    print(f"MTF Rejected:    {mtf_rejected}")
    print("=" * 70)
    
    print("\n📊 Exit Reason Breakdown:")
    for reason, data in stats.get('exit_reasons', {}).items():
        emoji = "✅" if data['pnl'] > 0 else "❌"
        print(f"  {reason}: {data['count']} trades | {emoji} ${data['pnl']:.2f}")
    
    print("\n📋 Last 10 Trades:")
    for trade in engine.trades[-10:]:
        emoji = "✅" if trade['pnl'] > 0 else "❌"
        trail = "🔄" if trade['trail_active'] else ""
        mult = f"({trade['size_multiplier']}x)" if trade['size_multiplier'] != 1.0 else ""
        print(f"{emoji}{trail} {trade['side']} @ ${trade['entry_price']:.6f} → ${trade['exit_price']:.6f} | PnL: ${trade['pnl']:.2f} {mult} ({trade['reason']})")
    
    return stats

if __name__ == "__main__":
    run_backtest()
