import subprocess
import json
import pandas as pd
from datetime import datetime, timezone
import sys

def fetch_table(table_name):
    query = f"SELECT * FROM {table_name}"
    # Double escaping quote hell workaround: just base64 encode the python script
    import base64
    py_script = f"""
import sqlite3, json
try:
    conn = sqlite3.connect('/data/trading.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute("{query}").fetchall()
    print(json.dumps([dict(r) for r in rows]))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
    b64_script = base64.b64encode(py_script.encode()).decode()
    cmd = [
        "fly", "ssh", "console", "-q", "-C",
        f"python3 -c \"import base64; exec(base64.b64decode('{b64_script}'))\""
    ]
    print(f"Fetching table {table_name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        # fly ssh console might print "Connecting to..." to stderr, but let's parse stdout
        for line in res.stdout.splitlines():
            if line.startswith('[') or line.startswith('{'):
                data = json.loads(line)
                if isinstance(data, dict) and "error" in data:
                    print(f"Error from remote: {data['error']}", file=sys.stderr)
                    return pd.DataFrame()
                return pd.DataFrame(data)
        
        # No JSON found
        print(f"No JSON found in output for {table_name}:\n{res.stdout}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error parsing {table_name}: {e}\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        return pd.DataFrame()

df_trades = fetch_table("trades")
df_closes = fetch_table("position_closes")

if df_trades.empty:
    print("Could not fetch trades table or it is empty.")
    sys.exit(0)

print("Trades shape:", df_trades.shape)
print("Closes shape:", df_closes.shape if not df_closes.empty else "Empty")

# Process trades
df_trades['open_time'] = pd.to_numeric(df_trades['open_time'], errors='coerce')
df_trades['close_time'] = pd.to_numeric(df_trades['close_time'], errors='coerce')

# Depending on if time is in seconds or ms
if df_trades['open_time'].max() > 1e11:
    df_trades['open_time_dt'] = pd.to_datetime(df_trades['open_time'], unit='ms', utc=True)
    df_trades['close_time_dt'] = pd.to_datetime(df_trades['close_time'], unit='ms', utc=True)
else:
    df_trades['open_time_dt'] = pd.to_datetime(df_trades['open_time'], unit='s', utc=True)
    df_trades['close_time_dt'] = pd.to_datetime(df_trades['close_time'], unit='s', utc=True)

today = datetime.now(timezone.utc).date()
today_trades = df_trades[(df_trades['open_time_dt'].dt.date == today) | (df_trades['close_time_dt'].dt.date == today)]

print("="*50)
print(f"📊 ANALYSIS FOR TODAY: {today}")
print("="*50)

print(f"Total trades found today: {len(today_trades)}")

if len(today_trades) > 0:
    print("\n1️⃣ Breakdown by Close Reason:")
    print(today_trades['reason'].value_counts())
    
    print("\n2️⃣ Analysis of Profitability:")
    df_trades['pnl_percent'] = pd.to_numeric(df_trades['pnl_percent'], errors='coerce')
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
        print(extreme_losses[['symbol', 'side', 'pnl_percent', 'reason', 'close_time_dt']])
    else:
        print("✅ No extreme losses found.")
        
    # 3.2 Missing reasons
    missing_reasons = today_trades[today_trades['reason'].isna() | (today_trades['reason'] == '') | (today_trades['reason'] == 'UNKNOWN')]
    if not missing_reasons.empty:
        print(f"⚠️ Found {len(missing_reasons)} trades with missing/UNKNOWN reasons!")
        print(missing_reasons[['symbol', 'pnl_percent', 'close_time_dt']])
    else:
        print("✅ All trades have mapped reasons.")
        
    # 3.3 VERY short duration trades (could be flash crashes, spread issues, or breakeven bugs)
    today_trades['duration_mins'] = (today_trades['close_time'] - today_trades['open_time']) / 60
    # adjust for ms if needed
    if today_trades['duration_mins'].max() > 1e3:
        today_trades['duration_mins'] = today_trades['duration_mins'] / 1000
        
    short_trades = today_trades[today_trades['duration_mins'] < 2.0]
    if not short_trades.empty:
        print(f"⚠️ Found {len(short_trades)} trades closed in under 2 minutes:")
        for _, row in short_trades.iterrows():
            print(f"   -> {row['symbol']} [{row['side']}] | Dur: {row['duration_mins']:.1f}m | Reason: {row['reason']} | PnL: {row['pnl_percent']}%")
    else:
        print("✅ No ultra-short duration trades found (good stability).")

    # 3.4 Cross-check with position_closes (if we merged it)
    if not df_closes.empty:
        print(f"\n4️⃣ Cross-checking with 'position_closes' table...")
        df_closes['close_time'] = pd.to_numeric(df_closes['timestamp'], errors='coerce')
        # matching
        orphans = today_trades[~today_trades['symbol'].isin(df_closes['symbol'])]
        print(f"Found detailed close logs for {len(today_trades) - len(orphans)} out of {len(today_trades)} today's trades.")

print("\nAnalysis complete.")
