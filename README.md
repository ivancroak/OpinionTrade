# OpinionTrade

A CLI trading bot for [Opinion.trade](https://opinion.trade) prediction markets on BNB Chain.

Places **pegged limit orders** that track the orderbook and stay N ticks below the best bid — avoiding adverse selection while capturing spread.

## Features

- **Pegged limit orders** — automatically adjusts price relative to best bid every 2 seconds
- **Market discovery** — filters and ranks markets by volume, open interest, and activity
- **Position liquidation** — monitors orderbook for break-even exit opportunities
- **Multi-wallet support** — run multiple wallets with independent proxy routing
- **SIWE authentication** — Sign-In With Ethereum, no API key needed
- **EIP-712 order signing** — on-chain order verification via typed data signatures
- **Configurable sizing** — fixed USDT amount or percentage of wallet balance

## How It Works

```
1. SELECT MARKET         Browse top markets by volume or lowest open interest
         ↓
2. CONFIGURE ORDER       Choose YES/NO side, tick offset (n), order size
         ↓
3. PEGGING LOOP          Every 2s: fetch orderbook → set price = best_bid - n × tick_size → place/adjust
         ↓
4. ON FILL               Close at market, place limit sell, or hold position
```

The tick offset `n` controls how far behind the best bid your order sits. With a tick size of $0.001, `n=10` means your order is always $0.01 below the best bid.

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
python run_bot.py                     # Interactive trading mode
python run_bot.py --test              # Test connection and wallet setup
python run_bot.py --liquidate         # Sell positions at break-even or better
python run_bot.py --verbose           # Enable debug logging
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
│   ├── strategy.py         # Pegged limit order strategy engine
│   ├── market_selection.py # Market filtering, ranking & display
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
- `opinion_clob_sdk` package (installed via requirements.txt)

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
