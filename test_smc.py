import pandas as pd
from smartmoneyconcepts import smc

data = {
    'open': [100, 105, 110, 120, 115, 110, 100, 95],
    'high': [105, 110, 120, 125, 120, 115, 105, 100],
    'low': [95, 100, 105, 115, 110, 105, 95, 90],
    'close': [104, 109, 119, 124, 111, 101, 96, 92],
    'volume': [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
}
df = pd.DataFrame(data)

swing_hl = smc.swing_highs_lows(df, swing_length=2)
print("Swing HL:\n", swing_hl)

ob = smc.ob(df, swing_highs_lows=swing_hl)
print("OB:\n", ob)

fvg = smc.fvg(df)
print("FVG:\n", fvg)

bos = smc.bos_choch(df, swing_hl)
print("BOS:\n", bos)
