import React, { useState } from 'react';
import { Wallet, Zap, BarChart3, Terminal, TrendingUp, TrendingDown, Bot } from 'lucide-react';

interface TabsProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    positionCount: number;
    signalCount: number;
    opportunitiesCount?: number;
    aiTrackingCount?: number;
    pendingConfirmedCount?: number; // Phase UI-Redesign: for dynamic badge accent
}

export const TabNavigation: React.FC<TabsProps> = ({
    activeTab,
    onTabChange,
    positionCount,
    signalCount,
    opportunitiesCount = 0,
    aiTrackingCount = 0,
    pendingConfirmedCount = 0
}) => {
    const tabs = [
        { id: 'portfolio', label: 'Portföy', icon: Wallet, badge: positionCount > 0 ? positionCount : null },
        { id: 'signals', label: 'Sinyaller', icon: Zap, badge: signalCount > 0 ? signalCount : null },
        { id: 'opportunities', label: 'Adaylar', icon: TrendingUp, badge: opportunitiesCount > 0 ? opportunitiesCount : null },
        { id: 'ai', label: 'Yapay Zeka', icon: Bot, badge: aiTrackingCount > 0 ? aiTrackingCount : null, color: 'fuchsia' },
        { id: 'performance', label: 'Performans', icon: BarChart3, badge: null },
        { id: 'logs', label: 'Loglar', icon: Terminal, badge: null },
    ];

    return (
        <div className="flex items-center gap-1 bg-[#0B0E14] border border-slate-800 rounded-xl p-1 mb-4 overflow-x-auto scrollbar-hide">
            {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;

                return (
                    <button
                        key={tab.id}
                        onClick={() => onTabChange(tab.id)}
                        className={`
              flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 sm:py-2.5 rounded-lg font-medium text-xs sm:text-sm transition-all whitespace-nowrap flex-shrink-0
              ${isActive
                                ? 'bg-indigo-600 text-white shadow-lg'
                                : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                            }
            `}
                    >
                        <Icon className="w-4 h-4" />
                        <span className="hidden sm:inline">{tab.label}</span>
                        {tab.badge && (
                            <span className={`
                text-[10px] font-bold px-1.5 py-0.5 rounded-full min-w-[18px] text-center
                ${isActive ? 'bg-white/20 text-white'
                                    : tab.id === 'signals' && pendingConfirmedCount > 0
                                        ? 'bg-cyan-500/20 text-cyan-400 animate-pulse'
                                        : tab.id === 'ai'
                                            ? 'bg-fuchsia-500/20 text-fuchsia-400'
                                            : 'bg-amber-500/20 text-amber-400'
                                }
              `}>
                                {tab.badge}
                            </span>
                        )}
                    </button>
                );
            })}
        </div>
    );
};

// Portfolio Tab Content
interface PortfolioTabProps {
    children: React.ReactNode;
}

export const PortfolioTab: React.FC<PortfolioTabProps> = ({ children }) => {
    return (
        <div className="space-y-4">
            {children}
        </div>
    );
};

// Signals Tab Content
interface SignalsTabProps {
    children: React.ReactNode;
}

export const SignalsTab: React.FC<SignalsTabProps> = ({ children }) => {
    return (
        <div className="space-y-4">
            {children}
        </div>
    );
};

// Opportunities Tab Content  
interface OpportunitiesTabProps {
    children: React.ReactNode;
}

export const OpportunitiesTab: React.FC<OpportunitiesTabProps> = ({ children }) => {
    return (
        <div className="space-y-4">
            {children}
        </div>
    );
};

// Logs Tab Content
interface LogsTabProps {
    children: React.ReactNode;
}

export const LogsTab: React.FC<LogsTabProps> = ({ children }) => {
    return (
        <div className="space-y-4">
            {children}
        </div>
    );
};
