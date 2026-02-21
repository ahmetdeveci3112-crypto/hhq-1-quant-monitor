import subprocess
import base64
import json

python_script = """
import sqlite3
import json
import traceback
import shutil
from datetime import datetime, timezone

try:
    # Copy DB to avoid OperationalError: database is locked
    shutil.copy2('/data/trading.db', '/tmp/trading_copy.db')
    
    conn = sqlite3.connect('/tmp/trading_copy.db')
    conn.row_factory = sqlite3.Row
    
    # SPEED OPTIMIZATION: Create index to make the query instant
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_sym_ts ON signals(symbol, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trades(open_time)")
    
    # Get today's start timestamp in seconds
    today_start_sec = int(datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    today_start_ms = today_start_sec * 1000
    
    # Try fetching trades from today directly using SQL
    # SQLite uses milliseconds or seconds?
    trades = conn.execute("SELECT * FROM trades WHERE open_time >= ? OR open_time >= ?", (today_start_sec, today_start_ms)).fetchall()
    
    results = []
    
    for t in trades:
        trade_dict = dict(t)
        symbol = trade_dict.get('symbol')
        open_time = trade_dict.get('open_time')
        
        # Try finding the exact signal that triggered this trade.
        # Assuming signal timestamp is close to trade open_time.
        # Window of 60 seconds (or 60000 ms)
        window = 60000 if open_time > 1e11 else 60
        
        signal = conn.execute(
            "SELECT * FROM signals WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (symbol, open_time - window, open_time + window)
        ).fetchone()
        
        if signal:
            sig_dict = dict(signal)
            trade_dict['signal_score'] = sig_dict.get('score')
            trade_dict['signal_telemetry'] = sig_dict.get('telemetry')
            
        # Try fetching position closes logic too
        p_close = conn.execute(
            "SELECT * FROM position_closes WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
            (symbol, trade_dict.get('close_time', 0) - window, trade_dict.get('close_time', 0) + window)
        ).fetchone()
        
        if p_close:
            pc_dict = dict(p_close)
            trade_dict['detailed_close_reason'] = pc_dict.get('reason')
            trade_dict['detailed_pnl'] = pc_dict.get('pnl_percent')
            
        results.append(trade_dict)
        
    print(json.dumps({'success': True, 'count': len(results), 'data': results}))

except Exception as e:
    print(json.dumps({'success': False, 'error': str(e), 'trace': traceback.format_exc()}))
"""

# Base64 encode the script to avoid character escaping issues in the bash shell
b64_script = base64.b64encode(python_script.encode('utf-8')).decode('utf-8')

cmd = [
    "fly", "ssh", "console", "-q", "-C", 
    f"python3 -c \"import base64; exec(base64.b64decode('{b64_script}'))\""
]

print("Executing rigorous analysis query on Fly.io machine...")
res = subprocess.run(cmd, capture_output=True, text=True)

try:
    # Filter fly outputs
    json_lines = [line for line in res.stdout.splitlines() if line.startswith('{')]
    if not json_lines:
        print("No JSON found in response.")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        exit(1)
        
    output = json.loads(json_lines[-1])
    
    if not output.get('success'):
        print("Error from server:", output.get('error'))
        print("Trace:", output.get('trace'))
        exit(1)
        
    trades = output['data']
    print(f"Successfully joined {len(trades)} trades with their signals.")
    
    # Analyze the trades with telemetry
    import pandas as pd
    if not trades:
        print("No trades found today.")
        exit(0)
        
    df = pd.DataFrame(trades)
    df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce')
    
    # Let's find losses < -3% and check their telemetry
    extreme_losses = df[df['pnl_percent'] < -3.0]
    
    print("\\n===========================================")
    print("🚨 EXTREME LOSSES ANALYSIS WITH TELEMETRY")
    print("===========================================")
    if extreme_losses.empty:
        print("No extreme losses found today!")
    else:
        for idx, row in extreme_losses.iterrows():
            reason = row.get('detailed_close_reason') or row.get('reason') or row.get('close_reason') or 'Unknown'
            print(f"\\n[{row['symbol']}] {row['side']} | PnL: {row['pnl_percent']}% | Reason: {reason}")
            # Try parsing telemetry
            telemetry_raw = row.get('signal_telemetry')
            if telemetry_raw:
                try:
                    tel = json.loads(telemetry_raw)
                    # Extract key info: leverage, strategy mode, etc.
                    strat = tel.get('strategy_mode', tel.get('strategyMode', 'UNKNOWN'))
                    lev = tel.get('leverage') or tel.get('dynamic_leverage', 'UNKNOWN')
                    smc = "Yes" if "OB" in str(tel.get('smc_zone', '')) else "No"
                    print(f"   -> Strategy: {strat} | Leverage: {lev}x | SMC Present: {smc}")
                    
                    if 'volatility' in tel:
                        print(f"   -> Volatility (ATR): {tel['volatility']}")
                    
                    # Print raw if we want to deep dive
                    # print(f"   -> RAW: {tel}")
                except Exception as e:
                    print(f"   -> Telemetry parse error: {e}")
            else:
                print("   -> NO SIGNAL TELEMETRY FOUND (Could be a manual or external entry)")

    print("\\n===========================================")
    print("⏱️ ULTRA-SHORT TRADES ANALYSIS")
    print("===========================================")
    df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
    df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
    
    df['duration_mins'] = (df['close_time'] - df['open_time']) / 60
    if df['duration_mins'].max() > 1e3:
        df['duration_mins'] = df['duration_mins'] / 1000
        
    short_trades = df[df['duration_mins'] < 2.0]
    if short_trades.empty:
        print("No ultra-short trades found.")
    else:
        for idx, row in short_trades.iterrows():
            reason = row.get('detailed_close_reason') or row.get('reason') or row.get('close_reason') or 'Unknown'
            print(f"\\n[{row['symbol']}] {row['side']} | Dur: {row['duration_mins']:.1f}m | PnL: {row['pnl_percent']}% | Reason: {reason}")
            telemetry_raw = row.get('signal_telemetry')
            if telemetry_raw:
                try:
                    tel = json.loads(telemetry_raw)
                    strat = tel.get('strategy_mode', tel.get('strategyMode', 'UNKNOWN'))
                    print(f"   -> Strategy: {strat} | Score: {row.get('signal_score', 'N/A')}")
                except:
                    pass

    print("\\n===========================================")
    print("📈 STRATEGY PERFORMANCE")
    print("===========================================")
    # Extract strategy from telemetry for all trades
    strategies = []
    for tel_str in df['signal_telemetry'].dropna():
        try:
            tel = json.loads(tel_str)
            strategies.append(tel.get('strategy_mode', tel.get('strategyMode', 'LEGACY')))
        except:
            strategies.append('UNKNOWN')
            
    if len(strategies) == len(df['signal_telemetry'].dropna()):
        # Just create a quick mask mapping if shapes align. Simplest way: loop over original df.
        pass
        
    df['strategy'] = 'UNKNOWN'
    for i in df.index:
        tel_str = df.at[i, 'signal_telemetry']
        if pd.notna(tel_str):
            try:
                tel = json.loads(tel_str)
                df.at[i, 'strategy'] = tel.get('strategy_mode', tel.get('strategyMode', 'LEGACY'))
            except:
                pass
                
    st_group = df.groupby('strategy')['pnl_percent'].agg(['count', 'mean', 'sum'])
    print(st_group)


except Exception as e:
    print("Error parsing wrapper:", str(e))
    import traceback
    traceback.print_exc()

