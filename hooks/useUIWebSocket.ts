import { useEffect, useRef, useCallback, useState } from 'react';

interface WebSocketMessage {
    type: string;
    data: any;
    timestamp: number;
}

interface UIWebSocketState {
    isConnected: boolean;
    lastMessage: WebSocketMessage | null;
    connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export function useUIWebSocket(
    wsUrl: string,
    onPositionUpdate?: (positions: any[]) => void,
    onSignal?: (signal: any) => void,
    onPositionOpened?: (position: any) => void,
    onPositionClosed?: (trade: any) => void,
    onKillSwitch?: (actions: any) => void,
    onLog?: (message: string) => void,
    onInitialState?: (state: any) => void
) {
    const [state, setState] = useState<UIWebSocketState>({
        isConnected: false,
        lastMessage: null,
        connectionStatus: 'disconnected'
    });

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setState(s => ({ ...s, connectionStatus: 'connecting' }));

        try {
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('🔌 UI WebSocket connected');
                setState(s => ({ ...s, isConnected: true, connectionStatus: 'connected' }));

                // Start ping interval
                pingIntervalRef.current = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('ping');
                    }
                }, 25000);
            };

            ws.onmessage = (event) => {
                // Handle ping/pong
                if (event.data === 'ping') {
                    ws.send('pong');
                    return;
                }
                if (event.data === 'pong') {
                    return;
                }

                try {
                    const message: WebSocketMessage = JSON.parse(event.data);
                    setState(s => ({ ...s, lastMessage: message }));

                    // Route message to appropriate handler
                    switch (message.type) {
                        case 'PRICE_UPDATE':
                            onPositionUpdate?.(message.data.positions);
                            break;
                        case 'SIGNAL':
                            onSignal?.(message.data);
                            break;
                        case 'POSITION_OPENED':
                            onPositionOpened?.(message.data);
                            break;
                        case 'POSITION_CLOSED':
                            onPositionClosed?.(message.data);
                            break;
                        case 'KILL_SWITCH':
                            onKillSwitch?.(message.data);
                            break;
                        case 'LOG':
                            onLog?.(message.data.message);
                            break;
                        case 'INITIAL_STATE':
                            onInitialState?.(message.data);
                            break;
                    }
                } catch (e) {
                    console.error('WebSocket message parse error:', e);
                }
            };

            ws.onclose = () => {
                console.log('🔌 UI WebSocket disconnected');
                setState(s => ({ ...s, isConnected: false, connectionStatus: 'disconnected' }));

                // Clear ping interval
                if (pingIntervalRef.current) {
                    clearInterval(pingIntervalRef.current);
                }

                // Auto-reconnect after 1 second (faster reconnection)
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('🔄 Attempting to reconnect...');
                    connect();
                }, 1000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setState(s => ({ ...s, connectionStatus: 'error' }));
            };

        } catch (error) {
            console.error('WebSocket connection error:', error);
            setState(s => ({ ...s, connectionStatus: 'error' }));
        }
    }, [wsUrl, onPositionUpdate, onSignal, onPositionOpened, onPositionClosed, onKillSwitch, onLog, onInitialState]);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setState(s => ({ ...s, isConnected: false, connectionStatus: 'disconnected' }));
    }, []);

    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    return {
        ...state,
        connect,
        disconnect
    };
}
