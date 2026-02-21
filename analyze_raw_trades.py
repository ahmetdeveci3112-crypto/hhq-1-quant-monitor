import json
import pandas as pd
from datetime import datetime, timezone

# Load raw_trades.json, filtering out any fly ssh connection logs
with open('raw_trades.json', 'r') as f:
    lines = f.readlines()

json_data = None
for line in lines:
    if line.startswith('[{'):
        json_data = json.loads(line)
        break

if not json_data:
    print("Failed to find JSON data in raw_trades.json.")
    exit(1)

df = pd.DataFrame(json_data)
print(f"Loaded {len(df)} total trades from DB.")

# Convert times
df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce')

if df['open_time'].max() > 1e11:
    df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df['close_time_dt'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
else:
    df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='s', utc=True)
    df['close_time_dt'] = pd.to_datetime(df['close_time'], unit='s', utc=True)

today = datetime.now(timezone.utc).date()
today_trades = df[(df['open_time_dt'].dt.date == today) | (df['close_time_dt'].dt.date == today)]

print("="*50)
print(f"📊 ANALYSIS FOR TODAY: {today}")
print("="*50)

print(f"Total trades found today: {len(today_trades)}")

if len(today_trades) > 0:
    reason_col = 'reason' if 'reason' in today_trades.columns else 'close_reason' if 'close_reason' in today_trades.columns else None
    
    if reason_col:
        print("\n1️⃣ Breakdown by Close Reason:")
        print(today_trades[reason_col].value_counts())
    else:
        print("\n1️⃣ No reason/close_reason column found.")
        print("Columns present:", today_trades.columns.tolist())
    
    print("\n2️⃣ Analysis of Profitability:")
    wins = today_trades[today_trades['pnl'] > 0]
    losses = today_trades[today_trades['pnl'] < 0]
    print(f"Wins: {len(wins)} | Losses: {len(losses)}")
    if len(wins) > 0: print(f"Avg Win: {wins['pnl_percent'].mean():.2f}%")
    if len(losses) > 0: print(f"Avg Loss: {losses['pnl_percent'].mean():.2f}%")
    
    # Inconsistencies check
    print("\n3️⃣ Inconsistencies and Anomalies:")
    
    # 3.1 Extreme losses (worse than emergency SL)
    extreme_losses = today_trades[today_trades['pnl_percent'] < -3.0]
    if not extreme_losses.empty:
        print(f"⚠️ Found {len(extreme_losses)} trades with loss > 3% (Check Emergency SL logic):")
        if reason_col:
            print(extreme_losses[['symbol', 'side', 'pnl_percent', reason_col, 'close_time_dt']])
        else:
            print(extreme_losses[['symbol', 'side', 'pnl_percent', 'close_time_dt']])
    else:
        print("✅ No extreme losses found.")
        
    # 3.2 Missing reasons
    if reason_col:
        missing_reasons = today_trades[today_trades[reason_col].isna() | (today_trades[reason_col] == '') | (today_trades[reason_col] == 'UNKNOWN')]
        if not missing_reasons.empty:
            print(f"⚠️ Found {len(missing_reasons)} trades with missing/UNKNOWN reasons!")
            print(missing_reasons[['symbol', 'pnl_percent', 'close_time_dt']])
        else:
            print("✅ All trades have mapped reasons.")
        
    # 3.3 VERY short duration trades (could be flash crashes, spread issues, or breakeven bugs)
    today_trades['duration_mins'] = (today_trades['close_time'] - today_trades['open_time']) / 60
    if today_trades['duration_mins'].max() > 1e3:
        today_trades['duration_mins'] = today_trades['duration_mins'] / 1000
        
    short_trades = today_trades[today_trades['duration_mins'] < 2.0]
    if not short_trades.empty:
        print(f"⚠️ Found {len(short_trades)} trades closed in under 2 minutes:")
        for _, row in short_trades.iterrows():
            reason_val = row[reason_col] if reason_col else 'N/A'
            print(f"   -> {row['symbol']} [{row['side']}] | Dur: {row['duration_mins']:.1f}m | Reason: {reason_val} | PnL: {row['pnl_percent']}%")
    else:
        print("✅ No ultra-short duration trades found (good stability).")

    # 3.4 Leverage/Size checks (if columns exist)
    if 'leverage' in today_trades.columns:
        high_leverage = today_trades[today_trades['leverage'] > 20]
        if not high_leverage.empty:
            print(f"⚠️ Found {len(high_leverage)} trades with extremely high leverage:")
            print(high_leverage[['symbol', 'leverage', 'pnl_percent', 'reason']])
