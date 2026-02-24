# Execution Data Integrity

## BBO Source Hierarchy

| Priority | Source | Key | Notes |
|----------|--------|-----|-------|
| 0 | **Cache** | `cache` | In-memory, TTL-gated (`EXEC_BBO_CACHE_TTL_MS`, default 400ms) |
| 1 | **REST bookTicker** | `bookTicker_rest` | `GET /fapi/v1/ticker/bookTicker?symbol=X` — reused `aiohttp.ClientSession` |
| 2 | **Order book top1** | `orderBook_top1` | `exchange.fetch_order_book(sym, 1)` via CCXT |
| — | **Invalid** | `none` | All sources failed or stale |

Future phase: `<symbol>@bookTicker` WebSocket → source key `bookTicker_ws`.

## BBO Config

| Env Var | Default | Description |
|---------|---------|-------------|
| `EXEC_BBO_MAX_AGE_MS` | 1500 | Max acceptable data age |
| `EXEC_BBO_RETRIES` | 3 | Retry attempts per fetch |
| `EXEC_BBO_RETRY_DELAY_MS` | 120 | Delay between retries |
| `EXEC_BBO_CACHE_TTL_MS` | 400 | Cache freshness window |
| `EXEC_BLOCK_ENTRY_ON_INVALID_BBO` | `true` | Block new entries on invalid BBO |
| `EXEC_ALLOW_CLOSE_WITH_INVALID_BBO` | `true` | Allow close with invalid BBO |

## Invalid BBO Policy

| Context | `allow=true` (default) | `allow=false` |
|---------|------------------------|---------------|
| **New entry** (market/limit) | ❌ Blocked (`NO_BBO_DATA`) | ❌ Blocked |
| **Pre-entry spread filter** | ❌ Vetoed (requeued) | ❌ Vetoed |
| **Close position** | ✅ **Proceeds** — telemetry shows NA | ❌ **Blocked** (`NO_BBO_DATA_CLOSE`) |

## EXEC_QUALITY Log Fields (parse-friendly)

```
📊 EXEC_QUALITY: LONG BTCUSDT MARKET
  | bid=$65000.123456 ask=$65001.234567
  | spread_pct=0.002%
  | fill=$65001.100000 slip_pct=+0.0002%
  | fee=$0.0300[order]
  | latency_ms=42
  | bbo_valid=True bbo_src=bookTicker_rest bbo_age_ms=23
```

| Field | Type | Notes |
|-------|------|-------|
| `bid` / `ask` | `$X.XXXXXX` or `NA` | Pre-trade BBO |
| `spread_pct` | `X.XXX%` or `NA` | BBO spread |
| `fill` | `$X.XXXXXX` | Actual fill price |
| `slip_pct` | `+X.XXXX%` or `NA` | Signed slippage vs BBO reference |
| `fee` | `$X.XXXX[source]` or `NA[source]` | Fee with source: `order`, `trades`, `unknown` |
| `latency_ms` | integer | Order send → response time |
| `bbo_valid` | `True` / `False` | BBO data validity |
| `bbo_src` | string | Source: `bookTicker_rest`, `orderBook_top1`, `cache`, `none` |
| `bbo_age_ms` | integer | BBO data age at order time |
| `bbo_reason` | string | Only present when `bbo_valid=False` |
| `trace` | string | Only for close operations with trace_id |

## Fee Fallback Chain

1. `order.fee.cost` → source=`order`
2. `fetch_my_trades(orderId=...)` → source=`trades`, note=`N_fills`
3. Neither → source=`unknown`, note describes reason

## Active/Pending BBO Focus

`should_refresh_bbo(symbol, active_symbols, pending_symbols)` returns `True` only for symbols with active positions or pending orders. Future use: gate WebSocket BBO subscriptions or reduce cache TTL for hot symbols.
