#!/usr/bin/env python3
"""
Opinion Trade Bot - Main Entry Point

Usage:
    python run_bot.py          # Run the trading bot
    python run_bot.py --test   # Test connection only
    python run_bot.py --help   # Show help

Before running:
1. Copy input_data/wallets.txt.example to input_data/wallets.txt
2. Add your private key(s) to wallets.txt
3. (Optional) Set up proxies in input_data/proxies.txt
4. Adjust settings in config.yaml if needed
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config, print_config_summary
from src.utils.logging_utils import setup_logger, get_logger
from src.opinion_client import OpinionClient
from src.bot import run_bot
from src.liquidator import run_liquidator


def test_connection(config):
    """
    Test mode: verify SDK connection and wallet setup.

    This helps debug issues before running the full bot.
    """
    logger = get_logger()

    print("\n" + "=" * 60)
    print("  CONNECTION TEST MODE")
    print("=" * 60)

    if not config.wallets:
        print("  ERROR: No wallets loaded!")
        print("  Make sure input_data/wallets.txt exists and has valid entries.")
        return False

    # Test first wallet
    wallet = config.wallets[0]
    print(f"\n  Testing wallet #{wallet.wallet_id}...")
    print(f"  Private key: {wallet.private_key[:6]}...{wallet.private_key[-4:]}")

    if wallet.has_proxy():
        print(f"  Proxy: {wallet.proxy}")
    else:
        print("  Proxy: None (direct connection)")

    # Try to connect
    client = OpinionClient(wallet, config.network)

    print("\n  Attempting to connect...")

    if not client.connect():
        print("  FAILED: Could not connect to Opinion.trade")
        print("\n  Troubleshooting:")
        print("  - Check your private key is correct")
        print("  - Make sure you have internet access")
        print("  - Try a different RPC URL in config.yaml")
        return False

    print(f"  SUCCESS: Connected!")
    print(f"  Wallet address: {client.address}")

    # Try to fetch some data
    print("\n  Fetching market data...")

    markets = client.get_markets(limit=3)

    if markets:
        print(f"  SUCCESS: Found {len(markets)} markets")
        print("\n  Sample markets:")
        for m in markets[:3]:
            print(f"    - {m.title}")
    else:
        print("  WARNING: Could not fetch markets (might be API issue)")

    # Try to fetch balance
    print("\n  Fetching wallet balance...")

    balance = client.get_balance()

    if balance:
        print(f"  SUCCESS: Balance fetched")
        print(f"    Available: ${balance.available:.2f}")
        print(f"    Frozen: ${balance.frozen:.2f}")
        print(f"    Total: ${balance.total:.2f}")
    else:
        print("  WARNING: Could not fetch balance")

    client.disconnect()

    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print("  Your setup appears to be working!")
    print("  Run without --test to start the trading bot.")
    print()

    return True


def run_liquidation_mode(config):
    """
    Liquidation mode: sell positions at break-even or better.

    This mode helps you exit positions without taking a loss by:
    - Monitoring the orderbook for favorable bids
    - Selling only when bids are at or above your average cost
    - Considering bid depth to avoid slippage
    """
    logger = get_logger()

    print("\n" + "=" * 60)
    print("   LIQUIDATION MODE")
    print("   Sell Positions at Break-Even or Better")
    print("=" * 60)

    if not config.wallets:
        print("  ERROR: No wallets loaded!")
        return

    # Select wallet (simplified - use first wallet or let user choose)
    if len(config.wallets) == 1:
        wallet = config.wallets[0]
        print(f"\n  Using wallet #{wallet.wallet_id}")
    else:
        print("\n  Available wallets:")
        for w in config.wallets:
            key_preview = f"{w.private_key[:6]}...{w.private_key[-4:]}"
            print(f"    {w.wallet_id}) {key_preview}")

        while True:
            choice = input("\n  Select wallet number (or 'q' to quit): ").strip().lower()
            if choice == 'q':
                return
            try:
                wallet_id = int(choice)
                wallet = config.get_wallet(wallet_id)
                if wallet:
                    break
                print("  Wallet not found.")
            except ValueError:
                print("  Invalid input.")

    # Connect
    print(f"\n  Connecting wallet #{wallet.wallet_id}...")
    client = OpinionClient(wallet, config.network)

    if not client.connect():
        print("  Failed to connect. Check your private key and network settings.")
        return

    print(f"  Connected! Address: {client.address}")

    try:
        # Run liquidator
        run_liquidator(client)
    finally:
        print("\n  Cleaning up...")
        client.disconnect()
        print("  Goodbye!")


def main():
    """Main entry point."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Opinion Trade Bot - Pegged Limit Order Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_bot.py          Run the trading bot
  python run_bot.py --test   Test connection only
  python run_bot.py -v       Verbose (debug) logging
        """
    )

    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test connection and exit'
    )

    parser.add_argument(
        '--liquidate', '-l',
        action='store_true',
        help='Run position liquidator (sell at break-even or better)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    args = parser.parse_args()

    # Set up logging first (with default settings)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    logger = get_logger()

    # Check that input_data folder exists
    if not Path("input_data").exists():
        print("ERROR: input_data folder not found!")
        print("Please create it and add your wallets.txt file.")
        sys.exit(1)

    # Check for wallets.txt
    if not Path("input_data/wallets.txt").exists():
        print("ERROR: input_data/wallets.txt not found!")
        print()
        print("To fix this:")
        print("1. Copy input_data/wallets.txt.example to input_data/wallets.txt")
        print("2. Edit wallets.txt and add your private key(s)")
        print()
        print("Format: wallet_number, private_key")
        print("Example: 1, 0xabcdef1234567890...")
        sys.exit(1)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Update logging with config settings
        setup_logger(
            level=log_level if args.verbose else config.logging.level,
            log_to_file=config.logging.log_to_file,
            log_file_prefix=config.logging.log_file_prefix
        )

        if args.test:
            # Test mode
            success = test_connection(config)
            sys.exit(0 if success else 1)
        elif args.liquidate:
            # Liquidation mode
            run_liquidation_mode(config)
        else:
            # Full bot mode
            run_bot(config)

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
