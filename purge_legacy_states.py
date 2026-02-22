import sqlite3
import re

def purge():
    conn = sqlite3.connect('/data/trading.db')
    c = conn.cursor()
    rows = c.execute('SELECT state_key FROM breakeven_states').fetchall()
    purged = 0
    for (k,) in rows:
        if re.match(r'^[A-Z0-9]+USDT_(LONG|SHORT)$', k):
            c.execute('DELETE FROM breakeven_states WHERE state_key=?', (k,))
            purged += 1
    conn.commit()
    print(f'Purged {purged} legacy breakeven states.')

if __name__ == "__main__":
    purge()
