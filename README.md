# OpinionTrade — Algorithmic Market-Making & Quant Trading Bot

> A Python algorithmic trading and market-making bot for **[Opinion.trade](https://opinion.trade)** CLOB prediction markets on **BNB Chain**. Combines a hand-rolled on-chain protocol integration (EIP-712 order signing, SIWE authentication, Gnosis-Safe multisig), a resilient real-time execution engine, and a quantitative / machine-learning layer that adaptively prices orders to maximize fill probability while minimizing adverse selection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Chain](https://img.shields.io/badge/Chain-BNB%20Chain%20(BSC)-f0b90b)
![License](https://img.shields.io/badge/License-MIT-green)

OpinionTrade places **pegged limit orders** that track the live orderbook and sit `N` ticks behind the best external bid — earning liquidity-provision rewards and capturing spread while staying out of the path of adverse selection. On top of that core loop sits a data-driven decision layer: the live orderbook is decomposed into quantitative microstructure features, and a machine-learning model selects order placement to maximize the probability of a favorable fill.

## Highlights

- **On-chain protocol, reimplemented from scratch** — EIP-712 typed-data order signing, a full SIWE (EIP-4361) login flow, and Gnosis-Safe multi-signature maker/signer separation, written directly against the exchange with no black-box SDK.
- **Quantitative market-microstructure analytics** — engineered orderbook features (depth imbalance, spread, liquidity concentration, volume/liquidity efficiency) computed across live markets to rank and select opportunities.
- **Machine-learning fill-probability model** — a scikit-learn model trained on historical orderbook snapshots predicts execution likelihood and adaptively tunes the tick offset, replacing static heuristics with a learned placement policy.
- **Resilient real-time execution engine** — external-best-bid pegging, partial-fill detection before re-pricing, an exponential-backoff retry state machine, a consecutive-failure circuit breaker, and an auto-flip recovery mode.
- **Break-even liquidation engine** — independently walks the orderbook to exit positions at break-even-or-better.
- **Money-correct by construction** — `Decimal` arithmetic end-to-end with deliberate tick-aligned truncation to match the exchange's on-chain integer semantics.
- **Multi-wallet + proxy routing**, **SIWE auth (no API keys)**, and **configurable position sizing**.

## Architecture

```
                          ┌────────────────────────────────────────────┐
                          │                 run_bot.py                  │
                          │        CLI · argparse · mode routing        │
                          └──────────────────┬─────────────────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              │                              │                              │
   ┌──────────▼──────────┐     ┌─────────────▼──────────┐     ┌─────────────▼────────┐
   │   Quant / ML Layer  │     │    Execution Engine    │     │  On-Chain Protocol   │
   │ ─────────────────── │     │ ────────────────────── │     │ ──────────────────── │
   │ orderbook feature   │     │ pegging strategy loop  │     │ SIWE (EIP-4361) auth │
   │ engineering         │────▶│ partial-fill detection │◀───▶│ EIP-712 order signing│
   │ microstructure      │     │ retry state machine    │     │ Gnosis-Safe multisig │
   │ signals + ranking   │     │ circuit breaker        │     │ wei / Decimal math   │
   │ scikit-learn        │     │ auto-flip recovery     │     │ Opinion.trade REST   │
   │ fill-prob model     │     │ liquidation engine     │     │ BNB Chain (BSC)      │
   └─────────────────────┘     └────────────────────────┘     └──────────────────────┘
```

Data flows top-down: the CLI loads config and wallets; the on-chain layer authenticates and exposes a typed client over the exchange; the quant/ML layer scores markets and chooses placement; and the execution engine drives the real-time order loop.

## Quantitative & Machine-Learning Layer

OpinionTrade treats order placement as a prediction problem rather than a fixed rule.

**Feature engineering.** For each candidate market and side, the live orderbook is decomposed into quantitative microstructure features:

- **Depth imbalance** — relative size of bid vs. ask liquidity.
- **Spread** — distance between best bid and best ask.
- **Liquidity concentration** — how stacked the book is near the top of book.
- **Efficiency indicator** — 24h traded volume relative to resting orderbook liquidity, surfacing high-activity / low-depth markets that fill quickly.

**Statistical ranking.** Markets are scored on these signals to prioritize opportunities with the best expected fill dynamics and the least adverse selection.

**Learned placement.** A scikit-learn model trained on historical orderbook snapshots estimates the **fill probability** of a pegged order at a given tick offset, and the engine selects the offset that maximizes expected liquidity capture — a data-driven policy that adapts to market conditions instead of a static "`N` ticks below best bid" constant.

## On-Chain Protocol Integration

Opinion.trade is a CLOB / conditional-token exchange on BNB Chain. OpinionTrade speaks its on-chain order protocol directly:

- **SIWE (EIP-4361) authentication** — builds and personal-signs a Sign-In-With-Ethereum message to obtain a session token; no API key, no custody.
- **EIP-712 typed-data order signing** — constructs the full `Order` struct (salt, maker, signer, taker, tokenId, maker/taker amounts, expiration, nonce, feeRateBps, side, signatureType) and signs it as EIP-712 typed data.
- **Gnosis-Safe multi-signature** — orders are signed by the EOA `signer` while the `maker` is the user's Safe proxy wallet, gated by an on-chain approval check before trading.
- **Wei-precise accounting** — all amounts use `Decimal` with deliberate tick-aligned truncation (not rounding) to match the exchange's integer tick semantics and avoid floating-point drift in financial code.

## Execution Engine

- **Pegging strategy** — every `poll_interval_seconds`, computes the best *external* bid (excluding the bot's own resting order) and places/re-prices a limit order exactly `n` ticks behind it.
- **Partial-fill safety** — detects partial fills *before* cancelling, so share accounting never desyncs.
- **Retry state machine** — HTTP client with exponential backoff and an exhaustive exception taxonomy (proxy / connection / timeout / empty-body / bad-JSON).
- **Circuit breaker** — aborts after consecutive failures, cancelling the live order first.
- **Auto-flip recovery** — if a buy fills unexpectedly, immediately places a sell at best ask and resumes.
- **Liquidation engine** — a separate mode that walks the bid side of the book to exit at break-even-or-better.

## Features

- **Pegged limit orders** — automatically adjusts price relative to the best external bid every 2 seconds
- **Quantitative market discovery** — filters and ranks markets by volume, open interest, activity, and microstructure efficiency signals
- **ML-assisted placement** — fill-probability model selects the tick offset that maximizes expected liquidity capture
- **Position liquidation** — monitors the orderbook for break-even exit opportunities
- **Multi-wallet support** — run multiple wallets with independent proxy routing
- **SIWE authentication** — Sign-In With Ethereum, no API key needed
- **EIP-712 order signing** — on-chain order verification via typed-data signatures
- **Configurable sizing** — fixed USDT amount or percentage of wallet balance

## How It Works

```
1. SELECT MARKET         Rank markets by volume / open interest / microstructure efficiency
         ↓
2. CONFIGURE ORDER       Choose YES/NO side, tick offset (n), order size
         ↓
3. PEGGING LOOP          Every 2s: fetch orderbook → score features → price = best_bid - n × tick_size → place/adjust
         ↓
4. ON FILL               Close at market, place limit sell (auto-flip), or hold position
```

The tick offset `n` controls how far behind the best bid your order sits. With a tick size of $0.001, `n=10` means your order is always $0.01 below the best bid — and the quant layer can adapt `n` to current book conditions.

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Language | Python 3.10+ |
| Quant / Data | pandas, NumPy |
| Machine Learning | scikit-learn |
| Blockchain | web3.py, eth-account, EIP-712, SIWE (EIP-4361), Gnosis Safe |
| Networking | requests, PySocks (SOCKS / HTTP proxy routing) |
| Exchange | Opinion.trade CLOB REST API · BNB Chain (BSC) |
| Tooling | PyYAML, colorama, rotating file logging |

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your wallet

```bash
cp input_data/wallets.txt.example input_data/wallets.txt
```

Edit `input_data/wallets.txt` and add your private key:

```
1, 0xYOUR_PRIVATE_KEY_HERE
```

### 3. (Optional) Configure proxies

```bash
cp input_data/proxies.txt.example input_data/proxies.txt
```

Edit with your proxy details — format: `wallet_number, ip:port:user:pass`

### 4. Test connection

```bash
python run_bot.py --test
```

### 5. Run

```bash
python run_bot.py
```

## Usage

```bash
python run_bot.py                       # Interactive trading mode
python run_bot.py --test                # Test connection and wallet setup
python run_bot.py --liquidate           # Sell positions at break-even or better
python run_bot.py --verbose             # Enable debug logging
python run_bot.py --config custom.yaml  # Use a custom config file
```

## Configuration

All settings are in `config.yaml`:

```yaml
network:
  chain_id: 56                          # BNB Chain
  rpc_url: "https://bsc-dataseed.binance.org"
  host: "https://proxy.opinion.trade:8443"

market_filters:
  min_24h_volume: 100                   # Minimum USDT volume to show a market
  top_n_by_volume: 10                   # Markets in "top by volume" list
  top_n_by_low_oi: 10                   # Markets in "lowest OI" list

trading:
  poll_interval_seconds: 2              # Orderbook refresh rate
  default_tick_offset: 10               # Ticks below best bid (10 = $0.01)
  default_order_size_usdt: 100          # Default order size
  max_open_orders: 1                    # Safety limit

order_size:
  mode: "fixed"                         # "fixed" or "percentage"
  fixed_amount: 100                     # USDT (when mode = fixed)
  percentage: 2                         # Wallet % (when mode = percentage)
```

## Project Structure

```
OpinionTrade/
├── run_bot.py              # Entry point & CLI argument parsing
├── config.yaml             # Bot configuration
├── requirements.txt        # Python dependencies
├── src/
│   ├── bot.py              # Session orchestration & interactive menus
│   ├── config_loader.py    # YAML config & wallet/proxy loading
│   ├── opinion_client.py   # Opinion.trade API client (SIWE auth, EIP-712 signing)
│   ├── strategy.py         # Pegged-order execution engine & adaptive placement
│   ├── market_selection.py # Market filtering, microstructure signals & ML-ranked selection
│   ├── liquidator.py       # Break-even position liquidation
│   └── utils/
│       └── logging_utils.py  # Colored console + rotating file logger
├── input_data/
│   ├── wallets.txt.example # Wallet file template
│   └── proxies.txt.example # Proxy file template
└── logs/                   # Auto-generated daily log files
```

## Prerequisites

- Python 3.10+
- BNB Chain wallet with USDT balance and BNB for gas
- Python scientific stack (`pandas`, `numpy`, `scikit-learn`) for the analytics / ML layer — installed via `requirements.txt`

## Security

- Private keys are stored locally in `input_data/wallets.txt` (git-ignored)
- The bot never logs or prints full private keys
- SIWE authentication — no API keys stored on external servers
- All order signing happens locally via EIP-712 typed data
- Use a dedicated wallet with limited funds

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Could not connect" | Verify private key (64 hex chars), check internet, try alternate RPC URL |
| "No markets found" | Lower `min_24h_volume` in config.yaml, check Opinion.trade status |
| "Failed to place order" | Ensure USDT balance on BSC, ensure BNB for gas, verify market is active |
| "Order not adjusting" | Check `poll_interval_seconds`, increase `--verbose` for debug logs |

## Disclaimer

This software interacts with real financial markets and trades real assets. Use at your own risk. The authors accept no responsibility for financial losses incurred through use of this bot. This is not financial advice.

## License

MIT

## Author

**Ivan Rykovski** — [GitHub](https://github.com/ivancroak) · [LinkedIn](https://www.linkedin.com/in/ivan-rykovski)

Developed privately during 2025–2026; published as a snapshot release in March 2026.
