"""
Phase 193: CCXT Pro WebSocket Manager
Auto-reconnect, heartbeat, fallback to REST, ticker/OHLCV cache.
"""
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import ccxt.pro for WebSocket support
try:
    import ccxt.pro as ccxtpro
    CCXT_PRO_AVAILABLE = True
except ImportError:
    CCXT_PRO_AVAILABLE = False
    logger.warning("⚠️ ccxt.pro not installed, WebSocket features disabled. Install with: pip install ccxt[pro]")


class CCXTWebSocketManager:
    """
    WebSocket bağlantı yöneticisi for CCXT Pro.
    - Auto-reconnect with exponential backoff
    - Heartbeat monitoring
    - Ticker/OHLCV cache (son bilinen değerler)
    - Fallback: WS fail → REST otomatik geçiş
    """
    
    def __init__(self, exchange_id: str = 'binance', sandbox: bool = False):
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        self.exchange = None
        self.running = False
        self.connected = False
        
        # Caches
        self.tickers: Dict[str, dict] = {}
        self.ohlcv_cache: Dict[str, list] = {}  # symbol -> [[timestamp, o, h, l, c, v], ...]
        
        # Timing
        self.last_ticker_update: float = 0
        self.last_ohlcv_update: Dict[str, float] = {}
        self.last_heartbeat: float = 0
        
        # Reconnect
        self._reconnect_delay = 1  # Start with 1s, max 60s
        self._max_reconnect_delay = 60
        
        # Callbacks
        self.on_ticker_update: Optional[Callable] = None
        self.on_ohlcv_update: Optional[Callable] = None
        
        # Stats
        self.ws_messages_received = 0
        self.ws_reconnect_count = 0
        self.fallback_to_rest = False
        
        # Subscribe symbols
        self._watch_symbols: Set[str] = set()
        self._watch_ohlcv_timeframes: Dict[str, str] = {}  # symbol -> timeframe
        
        logger.info(f"CCXTWebSocketManager initialized (exchange={exchange_id}, ccxt.pro={'✅' if CCXT_PRO_AVAILABLE else '❌'})")
    
    async def initialize(self, api_key: str = '', secret: str = '', **kwargs):
        """Initialize the exchange connection."""
        if not CCXT_PRO_AVAILABLE:
            logger.warning("ccxt.pro not available, WS manager disabled")
            self.fallback_to_rest = True
            return False
        
        try:
            exchange_class = getattr(ccxtpro, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'sandbox': self.sandbox,
                'options': {
                    'defaultType': 'future',
                    'watchTickers': {'maxSubscriptions': 300},
                },
                **kwargs
            })
            logger.info(f"✅ CCXT Pro exchange initialized: {self.exchange_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize CCXT Pro exchange: {e}")
            self.fallback_to_rest = True
            return False
    
    async def watch_tickers(self, symbols: list):
        """
        Watch real-time tickers for multiple symbols via WebSocket.
        Push-based: ~100ms latency vs 30s polling.
        """
        if not self.exchange or self.fallback_to_rest:
            return
        
        self._watch_symbols = set(symbols)
        self.running = True
        
        while self.running:
            try:
                # ccxt.pro watch_tickers returns instantly when new data arrives
                tickers = await self.exchange.watch_tickers(symbols)
                
                self.connected = True
                self._reconnect_delay = 1  # Reset backoff
                self.last_ticker_update = time.time()
                self.last_heartbeat = time.time()
                self.ws_messages_received += 1
                
                # Update cache
                for symbol, ticker in tickers.items():
                    self.tickers[symbol] = ticker
                
                # Callback
                if self.on_ticker_update:
                    try:
                        await self.on_ticker_update(tickers)
                    except Exception as e:
                        logger.warning(f"Ticker callback error: {e}")
                
            except Exception as e:
                self.connected = False
                self.ws_reconnect_count += 1
                logger.warning(f"WS ticker error (reconnect #{self.ws_reconnect_count}): {e}")
                
                # Exponential backoff
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '5m'):
        """
        Watch real-time OHLCV candles for a symbol.
        Updates come as each new candle forms.
        """
        if not self.exchange or self.fallback_to_rest:
            return
        
        self._watch_ohlcv_timeframes[symbol] = timeframe
        
        while self.running:
            try:
                ohlcv = await self.exchange.watch_ohlcv(symbol, timeframe)
                
                self.ohlcv_cache[symbol] = ohlcv
                self.last_ohlcv_update[symbol] = time.time()
                
                # Callback
                if self.on_ohlcv_update:
                    try:
                        await self.on_ohlcv_update(symbol, timeframe, ohlcv)
                    except Exception as e:
                        logger.warning(f"OHLCV callback error: {e}")
                
            except Exception as e:
                logger.warning(f"WS OHLCV error for {symbol}: {e}")
                await asyncio.sleep(5)
    
    def get_cached_ticker(self, symbol: str) -> Optional[dict]:
        """Get cached ticker, or None if not available."""
        ticker = self.tickers.get(symbol)
        if ticker and (time.time() - self.last_ticker_update) < 30:
            return ticker
        return None
    
    def get_cached_ohlcv(self, symbol: str) -> Optional[list]:
        """Get cached OHLCV, or None if stale."""
        ohlcv = self.ohlcv_cache.get(symbol)
        last_update = self.last_ohlcv_update.get(symbol, 0)
        if ohlcv and (time.time() - last_update) < 120:
            return ohlcv
        return None
    
    def is_healthy(self) -> bool:
        """Check if WS connection is healthy."""
        if self.fallback_to_rest:
            return False
        if not self.connected:
            return False
        if (time.time() - self.last_heartbeat) > 30:
            return False
        return True
    
    def get_status(self) -> dict:
        """Get WS manager status for monitoring."""
        return {
            'connected': self.connected,
            'fallback_to_rest': self.fallback_to_rest,
            'ccxt_pro_available': CCXT_PRO_AVAILABLE,
            'messages_received': self.ws_messages_received,
            'reconnect_count': self.ws_reconnect_count,
            'last_ticker_update': self.last_ticker_update,
            'cached_tickers': len(self.tickers),
            'cached_ohlcv': len(self.ohlcv_cache),
            'watched_symbols': len(self._watch_symbols),
            'healthy': self.is_healthy(),
        }
    
    async def close(self):
        """Cleanup and close connections."""
        self.running = False
        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing CCXT Pro exchange: {e}")
        logger.info("CCXTWebSocketManager closed")


# Global instance
ccxt_ws_manager = CCXTWebSocketManager()
