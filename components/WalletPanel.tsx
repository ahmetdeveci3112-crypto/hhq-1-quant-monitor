import React from 'react';
import { Wallet, TrendingUp, TrendingDown, Eye, EyeOff } from 'lucide-react';
import { Position } from '../types';

interface WalletPanelProps {
    walletBalance: number;
    unrealizedPnl: number;
    realizedPnl: number;  // From backend stats.totalPnl
    positions: Position[];
    initialBalance: number;
}

const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
        style: 'decimal',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    }).format(value);
};

export const WalletPanel: React.FC<WalletPanelProps> = ({
    walletBalance,
    unrealizedPnl,
    realizedPnl,
    positions,
    initialBalance = 10000,
}) => {
    // Wallet Balance = Initial Balance + Realized PnL (settled funds)
    const walletBalanceCalc = initialBalance + realizedPnl;

    // Margin Balance = Initial Balance + Unrealized PnL (total equity including unrealized)
    const marginBalance = initialBalance + unrealizedPnl;

    // Realized PnL percentage (from initial balance)
    const realizedPnlPercent = (realizedPnl / initialBalance) * 100;

    // Total Used Margin
    const usedMargin = positions.reduce((sum, p) => {
        const margin = (p as any).initialMargin || (p.sizeUsd || 0) / (p.leverage || 10);
        return sum + margin;
    }, 0);

    // Available Balance = Margin Balance - Used Margin
    const availableBalance = marginBalance - usedMargin;

    return (
        <div className="bg-[#0B0E14] rounded-2xl border border-slate-800 p-4 shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <h3 className="text-sm font-medium text-slate-400">Margin Balance</h3>
                    <Eye className="w-4 h-4 text-slate-600 cursor-pointer hover:text-slate-400" />
                </div>
                <div className="flex items-center gap-2 text-slate-500">
                    <Wallet className="w-4 h-4" />
                </div>
            </div>

            {/* Main Margin Balance */}
            <div className="mb-4">
                <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold text-white font-mono">
                        {formatCurrency(marginBalance)}
                    </span>
                    <span className="text-sm text-slate-400">USDT</span>
                </div>
                <div className="text-xs text-slate-500 mt-1">
                    ≈ ${formatCurrency(marginBalance)}
                </div>
            </div>

            {/* Today's Realized PnL */}
            <div className="mb-4">
                <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-400">Today's Realized PNL</span>
                    <span className={`text-sm font-medium ${realizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {realizedPnl >= 0 ? '+' : ''}{formatCurrency(realizedPnl)} ({realizedPnlPercent >= 0 ? '+' : ''}{realizedPnlPercent.toFixed(2)}%)
                    </span>
                </div>
            </div>

            {/* Wallet Balance & Unrealized PnL Row */}
            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <div className="text-xs text-slate-500 mb-1">Wallet Balance (USDT)</div>
                    <div className="text-base font-bold text-white font-mono">{formatCurrency(walletBalanceCalc)}</div>
                    <div className="text-[10px] text-slate-600">≈ ${formatCurrency(walletBalanceCalc)}</div>
                </div>
                <div>
                    <div className="text-xs text-slate-500 mb-1">Unrealized PNL (USDT)</div>
                    <div className={`text-base font-bold font-mono ${unrealizedPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {unrealizedPnl >= 0 ? '+' : ''}{formatCurrency(unrealizedPnl)}
                    </div>
                    <div className="text-[10px] text-slate-600">≈ ${formatCurrency(Math.abs(unrealizedPnl))}</div>
                </div>
            </div>

            {/* Available & Used Margin */}
            <div className="grid grid-cols-2 gap-4 pt-4 border-t border-slate-800">
                <div>
                    <div className="text-xs text-slate-500 mb-1">Available Balance</div>
                    <div className="text-sm font-bold text-cyan-400 font-mono">{formatCurrency(availableBalance)}</div>
                </div>
                <div>
                    <div className="text-xs text-slate-500 mb-1">Used Margin</div>
                    <div className="text-sm font-bold text-amber-400 font-mono">{formatCurrency(usedMargin)}</div>
                </div>
            </div>
        </div>
    );
};

// Enhanced Position Card with Binance Style
interface PositionCardBinanceProps {
    position: Position;
    currentPrice: number;
    onClose: () => void;
}

export const PositionCardBinance: React.FC<PositionCardBinanceProps> = ({
    position,
    currentPrice,
    onClose
}) => {
    const isLong = position.side === 'LONG';
    const pnl = position.unrealizedPnl || 0;
    const margin = (position as any).initialMargin || (position.sizeUsd || 0) / (position.leverage || 10);
    const roi = margin > 0 ? (pnl / margin) * 100 : 0;
    const sizeCoins = position.size || 0;
    const symbol = position.symbol.replace('USDT', '');

    return (
        <div className="bg-[#151921] rounded-xl border border-slate-800 p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <img
                        src={`https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/${symbol.toLowerCase()}.png`}
                        alt={symbol}
                        className="w-6 h-6"
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://raw.githubusercontent.com/spothq/cryptocurrency-icons/master/32/color/generic.png';
                        }}
                    />
                    <span className="font-bold text-white">{position.symbol}</span>
                    <span className="text-xs text-slate-500">Perp</span>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${isLong ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                        Cross {position.leverage || 10}X
                    </span>
                </div>
                <button
                    onClick={onClose}
                    className="text-xs text-rose-400 hover:text-rose-300 font-medium"
                >
                    × Kapat
                </button>
            </div>

            {/* PNL & ROI Row */}
            <div className="grid grid-cols-2 gap-4 mb-3">
                <div>
                    <div className="text-xs text-slate-500 mb-1">PNL (USDT)</div>
                    <div className={`text-lg font-bold font-mono ${pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
                    </div>
                </div>
                <div className="text-right">
                    <div className="text-xs text-slate-500 mb-1">ROI</div>
                    <div className={`text-lg font-bold font-mono ${roi >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {roi >= 0 ? '+' : ''}{roi.toFixed(2)}%
                    </div>
                </div>
            </div>

            {/* Size, Margin, Margin Ratio */}
            <div className="grid grid-cols-3 gap-3 mb-3">
                <div>
                    <div className="text-[10px] text-slate-500">Size ({symbol})</div>
                    <div className="text-xs font-mono text-white">{sizeCoins.toFixed(4)}</div>
                </div>
                <div>
                    <div className="text-[10px] text-slate-500">Margin (USDT)</div>
                    <div className="text-xs font-mono text-white">{margin.toFixed(2)}</div>
                </div>
                <div className="text-right">
                    <div className="text-[10px] text-slate-500">Margin Ratio</div>
                    <div className="text-xs font-mono text-white">
                        {((margin / (position.sizeUsd || 1)) * 100).toFixed(2)}%
                    </div>
                </div>
            </div>

            {/* Entry, Mark, Liq Price */}
            <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800">
                <div>
                    <div className="text-[10px] text-slate-500">Entry Price</div>
                    <div className="text-xs font-mono text-white">{position.entryPrice.toFixed(6)}</div>
                </div>
                <div>
                    <div className="text-[10px] text-slate-500">Mark Price</div>
                    <div className="text-xs font-mono text-white">{currentPrice.toFixed(6)}</div>
                </div>
                <div className="text-right">
                    <div className="text-[10px] text-slate-500">Liq.Price</div>
                    <div className="text-xs font-mono text-slate-600">--</div>
                </div>
            </div>
        </div>
    );
};
