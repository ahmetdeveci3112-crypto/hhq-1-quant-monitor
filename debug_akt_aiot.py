import subprocess
import base64
import json

python_script = """
import sqlite3
import json

try:
    conn = sqlite3.connect('/data/trading.db')
    conn.row_factory = sqlite3.Row
    
    closes = conn.execute(
        "SELECT symbol, side, reason, original_reason, entry_price, exit_price, pnl, timestamp FROM position_closes WHERE symbol IN ('AKTUSDT', 'AIOTUSDT') ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()
    
    print("=== RECENT CLOSES FOR AKT & AIOT ===")
    for c in closes:
        print(json.dumps(dict(c)))

    trades = conn.execute(
        "SELECT symbol, side, entry_price, exit_price, pnl_percent, open_time, close_time, close_reason, close_metrics_json FROM trades WHERE symbol IN ('AKTUSDT', 'AIOTUSDT') ORDER BY close_time DESC LIMIT 10"
    ).fetchall()
    
    print("\\n=== RECENT TRADES FOR AKT & AIOT ===")
    for t in trades:
        print(json.dumps(dict(t)))

    app_logs = conn.execute(
        "SELECT ts, time, message FROM logs WHERE message LIKE '%AKT%' OR message LIKE '%AIOT%' ORDER BY ts DESC LIMIT 100"
    ).fetchall()
    
    print("\\n=== RECENT SYSTEM LOGS FOR AKT & AIOT ===")
    for l in app_logs:
        print(dict(l))

except Exception as e:
    print("Error:", str(e))
"""

b64_script = base64.b64encode(python_script.encode('utf-8')).decode('utf-8')
cmd = ["fly", "ssh", "console", "-q", "-C", f"python3 -c \"import base64; exec(base64.b64decode('{b64_script}'))\""]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
