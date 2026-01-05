import asyncio
from main import run_backtest, BacktestRequest, logger
import logging

# Configure logger to print to stdout
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

async def main():
    request = BacktestRequest(
        symbol="DOGEUSDT",
        timeframe="15m",
        startDate="2025-12-01",
        endDate="2026-01-05",  # Updated to today
        initialBalance=10000,
        leverage=10,
        riskPerTrade=2
    )
    
    print("\n" + "="*50)
    print("  🐕 DOGEUSDT BACKTEST - Phase 29 Algorithm")
    print("  📊 Spread-Based Dynamic Leverage & Balance Protection")
    print("="*50 + "\n")
    
    print("⏳ Loading historical data...")
    
    try:
        result = await run_backtest(request)
        stats = result["stats"]
        trades = result.get("trades", [])
        
        print("\n" + "="*50)
        print("             BACKTEST SONUÇLARI               ")
        print("="*50)
        print(f"💰 Toplam PnL: ${stats['totalPnl']:.2f} ({stats['totalPnlPercent']:.1f}%)")
        print(f"📊 Kazanma Oranı: {stats['winRate']:.1f}%")
        print(f"🔢 Toplam İşlem: {stats['totalTrades']}")
        print(f"✅ Kazançlı: {stats['winningTrades']} | ❌ Zararlı: {stats['losingTrades']}")
        print(f"📉 Max Drawdown: {stats['maxDrawdown']:.2f}%")
        print(f"⚖️ Profit Factor: {stats['profitFactor']:.2f}")
        print(f"🏁 Son Bakiye: ${stats['finalBalance']:.2f}")
        print("="*50)
        
        # Show last 5 trades
        if trades:
            print("\n📝 Son 5 İşlem:")
            for trade in trades[-5:]:
                emoji = "✅" if trade.get('pnl', 0) > 0 else "❌"
                print(f"  {emoji} {trade.get('side', 'N/A')} | "
                      f"Entry: ${trade.get('entryPrice', 0):.6f} → "
                      f"Exit: ${trade.get('exitPrice', 0):.6f} | "
                      f"PnL: ${trade.get('pnl', 0):.2f} | "
                      f"{trade.get('closeReason', 'N/A')}")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
