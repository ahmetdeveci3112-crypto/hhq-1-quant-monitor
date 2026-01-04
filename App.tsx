import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Activity, Play, Square, Settings, Terminal, Wallet, ChevronDown, Network, AlertTriangle, Zap, BarChart3, Radio, Waves } from 'lucide-react';
import {
  MarketRegime, SystemState, TradeSignal, LiquidationEvent, OrderBookState,
  Portfolio, SystemSettings, Position, Trade, EquityPoint, PortfolioStats,
  BackendUpdate, BackendSignal, PendingOrder
} from './types';
import { formatPrice, formatCurrency } from './utils';
import { HurstPanel } from './components/HurstPanel';
import { PairsPanel } from './components/PairsPanel';
import { LiquidationPanel } from './components/LiquidationPanel';
import { OrderBookPanel } from './components/OrderBookPanel';
import { SettingsModal } from './components/SettingsModal';
import { PnLPanel } from './components/PnLPanel';
import { PositionPanel } from './components/PositionPanel';
import { BacktestPanel } from './components/BacktestPanel';
import { SMCPanel } from './components/SMCPanel';

// Backend WebSocket URL
// Backend WebSocket URL
// VITE_WS_URL will be provided by Vercel Environment Variables
const BACKEND_WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

const INITIAL_STATE: SystemState = {
  hurstExponent: 0.5,
  zScore: 0,
  spread: 120,
  atr: 0,
  activeLiquidationCascade: false,
  marketRegime: MarketRegime.RANDOM_WALK,
  currentPrice: 0,
};

const INITIAL_BALANCE = 10000;

const INITIAL_STATS: PortfolioStats = {
  totalTrades: 0,
  winningTrades: 0,
  losingTrades: 0,
  winRate: 0,
  totalPnl: 0,
  totalPnlPercent: 0,
  maxDrawdown: 0,
  profitFactor: 0,
  avgWin: 0,
  avgLoss: 0,
};

const COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT'];

// Helper for dynamic price formatting is now in utils.ts


// Map backend regime strings to MarketRegime enum
const parseRegime = (regime: string): MarketRegime => {
  switch (regime) {
    case 'TREND TAKİBİ':
      return MarketRegime.TREND_FOLLOWING;
    case 'ORTALAMAYA DÖNÜŞ':
      return MarketRegime.MEAN_REVERSION;
    default:
      return MarketRegime.RANDOM_WALK;
  }
};

// Generate unique ID
const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2);
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'backtest'>('live');
  const [isRunning, setIsRunning] = useState(false);
  const [selectedCoin, setSelectedCoin] = useState('BTCUSDT');
  const [systemState, setSystemState] = useState<SystemState>(INITIAL_STATE);
  const [logs, setLogs] = useState<string[]>([]);
  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [liquidations, setLiquidations] = useState<LiquidationEvent[]>([]);
  const [orderBook, setOrderBook] = useState<OrderBookState>({ bids: [], asks: [], imbalance: 0 });
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // Paper Trading State
  const [portfolio, setPortfolio] = useState<Portfolio>({
    balanceUsd: INITIAL_BALANCE,
    initialBalance: INITIAL_BALANCE,
    positions: [],
    trades: [],
    equityCurve: [{ time: Date.now(), balance: INITIAL_BALANCE, drawdown: 0 }],
    stats: INITIAL_STATS
  });

  // Pending Orders State (Pullback Entries)
  const [pendingOrders, setPendingOrders] = useState<PendingOrder[]>([]);

  // Settings Modal
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<SystemSettings>({
    leverage: 10,
    stopLossAtr: 2,
    takeProfit: 3,
    riskPerTrade: 2,
    trailActivationAtr: 1.5,
    trailDistanceAtr: 1,
    maxPositions: 1
  });

  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastSignalRef = useRef<BackendSignal | null>(null);

  const addLog = useCallback((msg: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [`[${timestamp}] ${msg}`, ...prev].slice(0, 100));
  }, []);

  // ============================================================================
  // PAPER TRADING ENGINE
  // ============================================================================

  const openPosition = useCallback((
    signal: BackendSignal,
    currentPrice: number
  ) => {
    setPortfolio(prev => {
      // Check if we already have a position
      if (prev.positions.length >= settings.maxPositions) {
        return prev;
      }

      // Apply confidence-based position sizing
      const sizeMultiplier = signal.sizeMultiplier ?? 1.0;
      const baseRiskAmount = prev.balanceUsd * (settings.riskPerTrade / 100);
      const adjustedRiskAmount = baseRiskAmount * sizeMultiplier;
      const positionSizeUsd = adjustedRiskAmount * settings.leverage;
      const positionSize = positionSizeUsd / currentPrice;

      const newPosition: Position = {
        id: generateId(),
        symbol: selectedCoin,
        side: signal.action,
        entryPrice: currentPrice,
        size: positionSize,
        sizeUsd: positionSizeUsd,
        stopLoss: signal.sl,
        takeProfit: signal.tp,
        trailingStop: signal.sl,
        trailActivation: signal.trailActivation,
        trailDistance: signal.trailDistance,
        isTrailingActive: false,
        unrealizedPnl: 0,
        unrealizedPnlPercent: 0,
        openTime: Date.now(),
        tp1Hit: false,
        sl1Hit: false
      };

      const confidenceText = signal.confidenceScore ? ` | Güven:${signal.confidenceScore}%` : '';
      const sizeText = sizeMultiplier !== 1.0 ? ` | Boyut:${sizeMultiplier}x` : '';
      addLog(`🚀 POZİSYON AÇILDI: ${signal.action} ${formatPrice(positionSize)} ${selectedCoin} (${formatCurrency(positionSizeUsd)}) @ $${formatPrice(currentPrice)} | SL:$${formatPrice(signal.sl)} TP:$${formatPrice(signal.tp)}${confidenceText}${sizeText}`);

      return {
        ...prev,
        positions: [...prev.positions, newPosition]
      };
    });
  }, [selectedCoin, settings, addLog]);

  const closePosition = useCallback((
    positionId: string,
    exitPrice: number,
    reason: 'SL' | 'TP' | 'TRAILING' | 'MANUAL' | 'SIGNAL' | 'TP1' | 'SL1',
    amountPercentage: number = 1.0
  ) => {
    setPortfolio(prev => {
      const position = prev.positions.find(p => p.id === positionId);
      if (!position) return prev;

      // Calculate PnL
      let pnl: number;
      if (position.side === 'LONG') {
        pnl = (exitPrice - position.entryPrice) * position.size;
      } else {
        pnl = (position.entryPrice - exitPrice) * position.size;
      }
      const pnlPercent = (pnl / position.sizeUsd) * 100 * settings.leverage;

      const newTrade: Trade = {
        id: generateId(),
        symbol: position.symbol,
        side: position.side,
        entryPrice: position.entryPrice,
        exitPrice: exitPrice,
        size: position.size,
        sizeUsd: position.sizeUsd,
        pnl: pnl,
        pnlPercent: pnlPercent,
        openTime: position.openTime,
        closeTime: Date.now(),
        closeReason: reason
      };

      const newBalance = prev.balanceUsd + pnl;
      const newTrades = [...prev.trades, newTrade];

      // Calculate stats
      const winningTrades = newTrades.filter(t => t.pnl > 0);
      const losingTrades = newTrades.filter(t => t.pnl <= 0);
      const totalWins = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
      const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));

      const newStats: PortfolioStats = {
        totalTrades: newTrades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        winRate: newTrades.length > 0 ? (winningTrades.length / newTrades.length) * 100 : 0,
        totalPnl: newBalance - prev.initialBalance,
        totalPnlPercent: ((newBalance - prev.initialBalance) / prev.initialBalance) * 100,
        maxDrawdown: Math.max(prev.stats.maxDrawdown, ((prev.initialBalance - newBalance) / prev.initialBalance) * 100),
        profitFactor: totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? 999 : 0,
        avgWin: winningTrades.length > 0 ? totalWins / winningTrades.length : 0,
        avgLoss: losingTrades.length > 0 ? totalLosses / losingTrades.length : 0
      };

      // Update equity curve
      const newEquityPoint: EquityPoint = {
        time: Date.now(),
        balance: newBalance,
        drawdown: newStats.maxDrawdown
      };

      const reasonText = reason === 'SL' ? 'STOP LOSS' : reason === 'TP' ? 'TAKE PROFIT' : reason === 'TRAILING' ? 'TRAILING STOP' : reason;
      const pnlText = pnl >= 0 ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`;
      const percentageText = amountPercentage < 1.0 ? ` (%${amountPercentage * 100})` : '';
      addLog(`${pnl >= 0 ? '✅' : '❌'} POZİSYON KAPANDI (${reasonText}${percentageText}): ${pnlText} (${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)`);

      let updatedPositions = prev.positions;
      if (amountPercentage < 1.0) {
        // Partial Close - Update Position
        updatedPositions = prev.positions.map(p => {
          if (p.id === positionId) {
            return {
              ...p,
              size: p.size * (1 - amountPercentage),
              sizeUsd: p.sizeUsd * (1 - amountPercentage),
              tp1Hit: reason === 'TP1' ? true : p.tp1Hit,
              sl1Hit: reason === 'SL1' ? true : p.sl1Hit
            };
          }
          return p;
        });
      } else {
        // Full Close - Remove Position
        updatedPositions = prev.positions.filter(p => p.id !== positionId);
      }

      return {
        ...prev,
        balanceUsd: newBalance,
        positions: updatedPositions,
        trades: newTrades,
        equityCurve: [...prev.equityCurve, newEquityPoint],
        stats: newStats
      };
    });
  }, [settings.leverage, addLog]);

  const updatePositions = useCallback((currentPrice: number) => {
    setPortfolio(prev => {
      if (prev.positions.length === 0) return prev;

      const updatedPositions = prev.positions.map(position => {
        // Calculate unrealized PnL
        let unrealizedPnl: number;
        if (position.side === 'LONG') {
          unrealizedPnl = (currentPrice - position.entryPrice) * position.size;
        } else {
          unrealizedPnl = (position.entryPrice - currentPrice) * position.size;
        }
        const unrealizedPnlPercent = (unrealizedPnl / position.sizeUsd) * 100 * settings.leverage;

        // Check for trailing stop activation
        let isTrailingActive = position.isTrailingActive;
        let trailingStop = position.trailingStop;
        let stopLoss = position.stopLoss;

        // BREAKEVEN TRIGGER (Profit Protection)
        // If profit > 0.5% and SL is not already at entry
        if (unrealizedPnlPercent > 0.5 && Math.abs(stopLoss - position.entryPrice) > 0.01 && !isTrailingActive) {
          stopLoss = position.entryPrice;
          addLog(`🛡️ BREAKEVEN TETİKLENDİ: Stop Girişe Çekildi (${position.entryPrice})`);
        }

        if (position.side === 'LONG') {
          if (currentPrice >= position.trailActivation && !isTrailingActive) {
            isTrailingActive = true;
            trailingStop = currentPrice - position.trailDistance;
            addLog(`📈 TRAILING STOP AKTİF: $${trailingStop.toFixed(2)}`);
          }
          if (isTrailingActive && currentPrice - position.trailDistance > trailingStop) {
            trailingStop = currentPrice - position.trailDistance;
          }
        } else {
          if (currentPrice <= position.trailActivation && !isTrailingActive) {
            isTrailingActive = true;
            trailingStop = currentPrice + position.trailDistance;
            addLog(`📉 TRAILING STOP AKTİF: $${trailingStop.toFixed(2)}`);
          }
          if (isTrailingActive && currentPrice + position.trailDistance < trailingStop) {
            trailingStop = currentPrice + position.trailDistance;
          }
        }

        return {
          ...position,
          unrealizedPnl,
          unrealizedPnlPercent,
          isTrailingActive,
          trailingStop,
          stopLoss: isTrailingActive ? trailingStop : stopLoss
        };
      });

      return { ...prev, positions: updatedPositions };
    });
  }, [settings.leverage, addLog]);

  const checkStopLossAndTakeProfit = useCallback((currentPrice: number) => {
    portfolio.positions.forEach(position => {
      // Calculate distances
      const tpDist = Math.abs(position.takeProfit - position.entryPrice);
      const slDist = Math.abs(position.stopLoss - position.entryPrice);

      // RESCUE MISSION Logic:
      // If position is open > 45 mins AND Price returns to entry (Breakeven)
      const duration = Date.now() - position.openTime;
      const isStale = duration > 45 * 60 * 1000; // 45 mins

      if (position.side === 'LONG') {
        const tp1Price = position.entryPrice + (tpDist * 0.5);
        const sl1Price = position.entryPrice - (slDist * 0.5);

        // Rescue Check
        if (isStale && currentPrice >= position.entryPrice) {
          addLog(`🚑 RESCUE MISSION: Uzun süreli pozisyon zararsız kapatıldı.`);
          closePosition(position.id, currentPrice, 'RESCUE');
          return;
        }

        // Check Full Stops
        if (currentPrice <= position.stopLoss) {
          closePosition(position.id, currentPrice, position.isTrailingActive ? 'TRAILING' : 'SL');
        } else if (currentPrice >= position.takeProfit) {
          closePosition(position.id, currentPrice, 'TP');
        }
        // Check Partial Stops (Scaling Out)
        else if (!position.tp1Hit && currentPrice >= tp1Price) {
          closePosition(position.id, currentPrice, 'TP1', 0.5);
        }
        else if (!position.sl1Hit && currentPrice <= sl1Price) {
          closePosition(position.id, currentPrice, 'SL1', 0.5);
        }

      } else {
        const tp1Price = position.entryPrice - (tpDist * 0.5);
        const sl1Price = position.entryPrice + (slDist * 0.5);

        // Rescue Check
        if (isStale && currentPrice <= position.entryPrice) {
          addLog(`🚑 RESCUE MISSION: Uzun süreli pozisyon zararsız kapatıldı.`);
          closePosition(position.id, currentPrice, 'RESCUE');
          return;
        }

        // Check Full Stops
        if (currentPrice >= position.stopLoss) {
          closePosition(position.id, currentPrice, position.isTrailingActive ? 'TRAILING' : 'SL');
        } else if (currentPrice <= position.takeProfit) {
          closePosition(position.id, currentPrice, 'TP');
        }
        // Check Partial Stops (Scaling Out)
        else if (!position.tp1Hit && currentPrice <= tp1Price) {
          closePosition(position.id, currentPrice, 'TP1', 0.5);
        }
        else if (!position.sl1Hit && currentPrice >= sl1Price) {
          closePosition(position.id, currentPrice, 'SL1', 0.5);
        }
      }
    });
  }, [portfolio.positions, closePosition]);

  const handleManualClose = useCallback((positionId: string) => {
    closePosition(positionId, systemState.currentPrice, 'MANUAL');
  }, [closePosition, systemState.currentPrice]);

  const checkPendingOrders = useCallback((currentPrice: number) => {
    // Check if any pending orders triggered
    if (pendingOrders.length === 0) return;

    setPendingOrders(prev => {
      const remaining: PendingOrder[] = [];
      const triggered: PendingOrder[] = [];

      prev.forEach(order => {
        const age = Date.now() - order.timestamp;

        // Timeout (15 mins)
        if (age > 15 * 60 * 1000) {
          addLog(`⏰ Bekleyen emir iptal (Zaman aşımı): ${order.symbol}`);
          return;
        }

        // Check trigger condition
        // LONG: Price was above, drops to limit
        // SHORT: Price was below, rises to limit
        // Ideally we check if price crossed currentPrice vs previousPrice, but here we just check bounds
        // to simplify: if Price is better than or equal to Limit for entry?
        // Actually for Limit Buy (Long), Price <= Entry. For Limit Sell (Short), Price >= Entry.

        let triggeredOrder = false;
        if (order.side === 'LONG' && currentPrice <= order.entryPrice) {
          triggeredOrder = true;
        } else if (order.side === 'SHORT' && currentPrice >= order.entryPrice) {
          triggeredOrder = true;
        }

        if (triggeredOrder) {
          triggered.push(order);
        } else {
          remaining.push(order);
        }
      });

      // Execute triggered
      triggered.forEach(order => {
        addLog(`⚡ Bekleyen Emir Tetiklendi: ${order.symbol} @ ${order.entryPrice}`);
        // Construct BackendSignal-like object to reuse openPosition logic
        const signal: BackendSignal = {
          action: order.side,
          entry: order.entryPrice,
          sl: order.sl,
          tp: order.tp,
          trailActivation: order.trailActivation,
          trailDistance: order.trailDistance,
          reason: "PULLBACK ENTRY",
          timestamp: Date.now(),
          confidence: 'HIGH',
          sizeMultiplier: order.sizeMultiplier,
          price: currentPrice
        };
        openPosition(signal, currentPrice);
      });

      return remaining;
    });
  }, [pendingOrders, openPosition, addLog]);

  const handleSignal = useCallback((signal: BackendSignal) => {
    // Instead of opening immediately, create a Pending Limit Order (Pullback Entry)
    const newOrder: PendingOrder = {
      id: generateId(),
      symbol: selectedCoin,
      side: signal.action,
      entryPrice: signal.entryPrice || signal.entry, // Use Pullback Price if available
      signalPrice: signal.price || signal.entry,
      sl: signal.sl,
      tp: signal.tp,
      trailActivation: signal.trailActivation,
      trailDistance: signal.trailDistance,
      sizeMultiplier: signal.sizeMultiplier || 1.0,
      timestamp: Date.now(),
      reason: signal.reason
    };

    setPendingOrders(prev => [...prev, newOrder]);
    addLog(`⏳ BEKLEYEN EMİR: ${signal.action} @ ${newOrder.entryPrice.toFixed(2)} (Sinyal: ${signal.price?.toFixed(2)})`);
  }, [selectedCoin, addLog]);

  // ============================================================================
  // WEBSOCKET CONNECTION
  // ============================================================================

  // Ref to store latest handlers to avoid WebSocket reconnection on state changes
  const wsHandlersRef = useRef({
    addLog,
    openPosition,
    handleSignal,
    checkPendingOrders,
    updatePositions,
    checkStopLossAndTakeProfit,
    portfolio,
    lastSignal: lastSignalRef.current
  });

  // Update ref on every render
  useEffect(() => {
    wsHandlersRef.current = {
      addLog,
      openPosition,
      handleSignal,
      checkPendingOrders,
      updatePositions,
      checkStopLossAndTakeProfit,
      portfolio,
      lastSignal: lastSignalRef.current
    };
  }, [addLog, openPosition, handleSignal, checkPendingOrders, updatePositions, checkStopLossAndTakeProfit, portfolio]);

  // ============================================================================
  // WEBSOCKET CONNECTION
  // ============================================================================

  useEffect(() => {
    if (!isRunning) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      return;
    }

    const connectWebSocket = () => {
      // Cleanup existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      const wsUrl = `${BACKEND_WS_URL}?symbol=${selectedCoin}`;
      wsHandlersRef.current.addLog(`Bağlanıyor: ${wsUrl}`);
      setConnectionError(null);

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsHandlersRef.current.addLog("🟢 Python Backend Bağlandı (HHQ-1 v2.0)");
        setConnectionError(null);
      };

      ws.onmessage = (event) => {
        const handlers = wsHandlersRef.current;
        try {
          const data: BackendUpdate = JSON.parse(event.data);

          if (data.type === 'update') {
            const { price, metrics, orderBook: ob, liquidation, signal } = data;

            // Update system state
            setSystemState({
              currentPrice: price,
              hurstExponent: metrics.hurst,
              marketRegime: parseRegime(metrics.regime),
              zScore: metrics.zScore,
              spread: metrics.spread,
              atr: metrics.atr,
              activeLiquidationCascade: !!liquidation?.isCascade,
              whaleZ: metrics.whale_z,
              smc: data.smc,
              pivots: data.pivots
            });

            // Update order book
            if (ob) {
              setOrderBook({
                bids: ob.bids || [],
                asks: ob.asks || [],
                imbalance: ob.imbalance || 0
              });
            }

            // Handle liquidation events
            if (liquidation) {
              setLiquidations(prev => [{
                id: generateId(),
                symbol: selectedCoin,
                side: liquidation.side === 'SELL' ? 'SATIM' : 'ALIM',
                amountUsd: liquidation.amount,
                price: liquidation.price,
                timestamp: Date.now(),
                isReal: true,
                isCascade: liquidation.isCascade
              }, ...prev].slice(0, 50));

              if (liquidation.isCascade) {
                handlers.addLog(`🔥 LİKİDASYON CASCADE: $${(liquidation.amount / 1000).toFixed(0)}k @ ${formatPrice(liquidation.price)}`);
              }
            }

            // Handle trading signals from backend
            if (signal && signal.timestamp !== lastSignalRef.current?.timestamp) {
              lastSignalRef.current = signal;

              // Add to signals list
              setSignals(prev => [{
                id: generateId(),
                timestamp: Date.now(),
                pair: selectedCoin,
                type: signal.action === 'LONG' ? 'UZUN (LONG)' : 'KISA (SHORT)',
                reason: signal.reason,
                status: 'İŞLENDİ',
                price: price
              }, ...prev].slice(0, 20));

              // Open position if we don't have one and not at max pos
              if (handlers.portfolio.positions.length === 0) {
                // OLD: handlers.openPosition(signal, price);
                // NEW: Use Handle Signal to create Pending Order logic
                handlers.handleSignal(signal);
              }
            }

            // Update positions with current price
            handlers.updatePositions(price);

            // Check Pending Orders (Limit Entries)
            handlers.checkPendingOrders(price);

            // Check for SL/TP hits
            handlers.checkStopLossAndTakeProfit(price);
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      };

      ws.onclose = () => {
        wsHandlersRef.current.addLog("🔴 Bağlantı kesildi.");
        if (isRunning) {
          // wsHandlersRef.current.addLog("🔄 3 saniye sonra yeniden bağlanılacak...");
          reconnectTimeoutRef.current = setTimeout(() => {
            // Check ref directly
            if (wsRef.current) return;
            connectWebSocket();
          }, 3000);
        }
      };

      ws.onerror = () => {
        setConnectionError("Backend'e bağlanılamadı. Python sunucusunun çalıştığından emin olun.");
        wsHandlersRef.current.addLog("🔴 Bağlantı Hatası! Backend aktif mi kontrol edin.");
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [isRunning, selectedCoin]);

  // Periodic equity curve update
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setPortfolio(prev => {
        const lastPoint = prev.equityCurve[prev.equityCurve.length - 1];
        const currentEquity = prev.balanceUsd +
          prev.positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);

        // Only add new point if significant change or every minute
        if (Math.abs(currentEquity - lastPoint.balance) > 10 ||
          Date.now() - lastPoint.time > 60000) {
          return {
            ...prev,
            equityCurve: [...prev.equityCurve, {
              time: Date.now(),
              balance: currentEquity,
              drawdown: prev.stats.maxDrawdown
            }].slice(-500) // Keep last 500 points
          };
        }
        return prev;
      });
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);


  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} settings={settings} onSave={setSettings} />}

      {/* Connection Error Banner */}
      {connectionError && (
        <div className="bg-red-500/10 border-b border-red-500/20 px-4 py-3 flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400" />
          <span className="text-red-400 text-sm">{connectionError}</span>
          <span className="text-red-400/60 text-xs ml-auto">
            Çalıştır: <code className="bg-slate-800 px-2 py-0.5 rounded">python3 main.py</code>
          </span>
        </div>
      )}

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className={`p-2 rounded-lg transition-colors ${isRunning ? 'bg-indigo-500/20 text-indigo-400' : 'bg-slate-800 text-slate-400'}`}>
              <Activity className="w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight text-white">HHQ-1 Quant Monitor <span className="text-indigo-400">v2.0</span></h1>
              <div className="flex items-center gap-2 text-xs">
                <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                  <span className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
                  <span className="text-slate-400 font-mono">{isRunning ? 'CANLI' : 'DURDURULDU'}</span>
                </div>
                {isRunning && (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-purple-400">
                    <Network className="w-3 h-3" />
                    <span>4-LAYER ALGO</span>
                  </div>
                )}
                {portfolio.positions.length > 0 && (
                  <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400 animate-pulse">
                    <Zap className="w-3 h-3" />
                    <span>POZİSYON AKTİF</span>
                  </div>
                )}
              </div>
            </div>

            {/* Coin Selector */}
            <div className="hidden md:flex ml-6 relative group">
              <button className="flex items-center gap-2 bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg transition-colors border border-slate-700">
                <img src={`https://lcw.nyc3.cdn.digitaloceanspaces.com/production/currencies/64/${selectedCoin.replace('USDT', '').toLowerCase()}.png`} className="w-5 h-5 rounded-full" onError={(e) => e.currentTarget.style.display = 'none'} />
                <span className="font-bold">{selectedCoin}</span>
                <ChevronDown className="w-4 h-4 text-slate-400" />
              </button>
              <div className="absolute top-full left-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50">
                {COINS.map(c => (
                  <button
                    key={c}
                    onClick={() => { setSelectedCoin(c); setIsRunning(false); setTimeout(() => setIsRunning(true), 100); }}
                    className="w-full text-left px-4 py-3 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg text-sm font-medium"
                  >
                    {c}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Paper Trading Balance */}
            <div className="hidden lg:flex items-center gap-3 px-4 py-2 bg-slate-900 rounded-lg border border-slate-800">
              <div className={`p-1.5 rounded ${portfolio.balanceUsd >= INITIAL_BALANCE ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                <Wallet className="w-4 h-4" />
              </div>
              <div className="flex flex-col items-end leading-none">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Bakiye</span>
                <span className={`font-mono font-bold ${portfolio.balanceUsd >= INITIAL_BALANCE ? 'text-emerald-400' : 'text-red-400'}`}>
                  ${portfolio.balanceUsd.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                </span>
              </div>
            </div>

            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md font-medium transition-all ${isRunning
                ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20'
                : 'bg-emerald-500 text-slate-900 hover:bg-emerald-400 shadow-lg shadow-emerald-500/20'
                }`}
            >
              {isRunning ? <><Square className="w-4 h-4 fill-current" /> DURDUR</> : <><Play className="w-4 h-4 fill-current" /> BAŞLAT</>}
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-slate-800 bg-slate-900/30">
        <div className="max-w-[1800px] mx-auto px-4">
          <div className="flex items-center gap-1">
            <button
              onClick={() => setActiveTab('live')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'live'
                ? 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
            >
              <Radio className="w-4 h-4" />
              Canlı Trading
            </button>
            <button
              onClick={() => setActiveTab('backtest')}
              className={`flex items-center gap-2 px-5 py-3 text-sm font-medium border-b-2 transition-colors ${activeTab === 'backtest'
                ? 'border-indigo-500 text-indigo-400'
                : 'border-transparent text-slate-500 hover:text-slate-300'
                }`}
            >
              <BarChart3 className="w-4 h-4" />
              Backtest
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {activeTab === 'backtest' ? (
        <main className="max-w-[1800px] mx-auto px-4 py-6">
          <BacktestPanel selectedCoin={selectedCoin} />
        </main>
      ) : (
        <main className="max-w-[1800px] mx-auto px-4 py-6 grid grid-cols-12 gap-4">

          {/* Left Column - Analytics */}
          <div className="col-span-12 lg:col-span-6 xl:col-span-5 grid grid-cols-12 gap-4 h-fit">

            {/* Layer 1: Market Regime */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <HurstPanel hurst={systemState.hurstExponent} regime={systemState.marketRegime} />
            </div>

            {/* Layer 2: Z-Score & Spread */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <PairsPanel zScore={systemState.zScore} spread={systemState.spread} />
            </div>

            {/* Layer 3: Liquidation */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <LiquidationPanel events={liquidations} active={systemState.activeLiquidationCascade} />
            </div>

            {/* Layer 4: Order Book */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <OrderBookPanel data={orderBook} />
            </div>


            {/* Layer 5: Whale Hunter */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-full relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                  <Waves className="w-16 h-16" />
                </div>
                <div className="flex items-center gap-2 mb-3">
                  <Waves className="w-5 h-5 text-blue-400" />
                  <h3 className="text-sm font-bold text-slate-300">BALİNA AVCISI</h3>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-500">Whale Z-Score</span>
                      <span className={systemState.whaleZ && systemState.whaleZ > 2 ? 'text-emerald-400' : systemState.whaleZ && systemState.whaleZ < -2 ? 'text-red-400' : 'text-slate-400'}>
                        {systemState.whaleZ?.toFixed(2) || '0.00'}
                      </span>
                    </div>
                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all duration-500 ${(systemState.whaleZ || 0) > 0 ? 'bg-emerald-500' : 'bg-red-500'
                          }`}
                        style={{ width: `${Math.min(Math.abs(systemState.whaleZ || 0) * 20, 100)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Layer 6: SMC Panel (Phase 10) */}
            <div className="col-span-12 md:col-span-6 lg:col-span-12 xl:col-span-6">
              <SMCPanel
                data={systemState.smc}
                pivots={systemState.pivots}
                currentPrice={systemState.currentPrice}
              />
            </div>
          </div>

          {/* Middle Column - Trading */}
          <div className="col-span-12 lg:col-span-3 xl:col-span-4 flex flex-col gap-4">
            {/* Price Box */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 text-center">
              <span className="text-slate-500 text-xs uppercase tracking-widest block mb-2">{selectedCoin} FİYAT</span>
              <div className={`text-4xl font-mono font-bold ${systemState.currentPrice > 0 ? 'text-white' : 'text-slate-600'}`}>
                ${formatPrice(systemState.currentPrice)}
              </div>
              {systemState.atr > 0 && (
                <div className="flex justify-center items-center gap-2 mt-2">
                  <span className="text-[10px] text-slate-500 font-mono">ATR: ${systemState.atr.toFixed(2)}</span>
                </div>
              )}
            </div>

            {/* Active Position */}
            <PositionPanel
              positions={portfolio.positions}
              currentPrice={systemState.currentPrice}
              onClosePosition={handleManualClose}
            />

            {/* PnL Chart */}
            <PnLPanel
              equityCurve={portfolio.equityCurve}
              stats={portfolio.stats}
              currentBalance={portfolio.balanceUsd}
              initialBalance={portfolio.initialBalance}
            />
          </div>

          {/* Right Column - Logs & Signals */}
          <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
            {/* Console */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col flex-1 shadow-inner max-h-[400px]">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center gap-2 bg-slate-900/50">
                <Terminal className="w-4 h-4 text-slate-400" />
                <h3 className="font-semibold text-slate-300 text-sm">Sistem Günlükleri</h3>
              </div>
              <div className="flex-1 overflow-y-auto p-4 font-mono text-[11px] space-y-2" ref={logRef}>
                {logs.length === 0 && (
                  <div className="flex flex-col items-center justify-center h-full text-slate-600 space-y-2">
                    <p>Sistem Hazır.</p>
                    <p className="text-slate-700">Başlat butonuna basın...</p>
                  </div>
                )}
                {logs.map((log, i) => (
                  <div key={i} className={`pb-1 border-b border-slate-800/50 last:border-0 break-words ${log.includes('POZİSYON AÇILDI') ? 'text-indigo-400 font-bold' :
                    log.includes('POZİSYON KAPANDI') && log.includes('✅') ? 'text-emerald-400 font-bold' :
                      log.includes('POZİSYON KAPANDI') && log.includes('❌') ? 'text-red-400 font-bold' :
                        log.includes('TRAILING') ? 'text-amber-400' :
                          log.includes('LİKİDASYON') ? 'text-orange-500 font-bold' :
                            'text-slate-400'
                    }`}>
                    <span className="opacity-50 mr-2">{log.split(']')[0]}]</span>
                    <span>{log.split(']')[1]}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Signals List */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col max-h-[300px]">
              <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <h3 className="font-semibold text-slate-300 text-sm">Sinyaller</h3>
                <span className="text-xs text-slate-500">{signals.length} adet</span>
              </div>
              <div className="overflow-y-auto p-2 space-y-2">
                {signals.map(s => (
                  <div key={s.id} className={`p-3 bg-slate-800/50 rounded border-l-2 ${s.type.includes('LONG') ? 'border-emerald-500' : 'border-red-500'}`}>
                    <div className="flex justify-between items-center mb-1">
                      <span className={`font-bold text-xs ${s.type.includes('LONG') ? 'text-emerald-400' : 'text-red-400'}`}>{s.type}</span>
                      <span className="text-[10px] text-slate-500">{new Date(s.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-white font-mono">${formatPrice(s.price)}</span>
                    </div>
                    <div className="text-[9px] text-slate-500 mt-1">{s.reason}</div>
                  </div>
                ))}
                {signals.length === 0 && <div className="text-center text-slate-600 text-xs py-4">Sinyal bekleniyor...</div>}
              </div>
            </div>
          </div>

        </main>
      )
      }
    </div >
  );
}