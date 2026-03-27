"""
Bot orchestration module.

This is the main controller that:
- Manages the overall flow
- Handles user interaction menus
- Coordinates market selection, order setup, and strategy execution
"""

from decimal import Decimal
from typing import Optional

from .config_loader import BotConfig, WalletConfig, print_config_summary
from .opinion_client import OpinionClient, OpinionClientManager
from .market_selection import (
    select_market_interactive,
    select_side_interactive
)
from .strategy import (
    StrategyState,
    get_token_id_for_side,
    run_pegging_loop
)
from .liquidator import run_liquidator
from .utils.logging_utils import get_logger, log_wallet


def show_welcome_banner():
    """Display welcome message."""
    print()
    print("=" * 60)
    print("   OPINION TRADE BOT v1.0")
    print("   Pegged Limit Order Strategy")
    print("=" * 60)
    print()


def select_wallet_interactive(config: BotConfig) -> Optional[WalletConfig]:
    """
    Let user select which wallet to use (if multiple).

    Args:
        config: Bot configuration with wallet list

    Returns:
        Selected wallet config, or None if cancelled
    """
    wallets = config.wallets

    if len(wallets) == 1:
        # Only one wallet, use it automatically
        print(f"  Using wallet #{wallets[0].wallet_id}")
        return wallets[0]

    print("\n" + "-" * 40)
    print("  SELECT WALLET")
    print("-" * 40)
    print("  Available wallets:")

    for w in wallets:
        key_preview = f"{w.private_key[:6]}...{w.private_key[-4:]}"
        proxy_info = f" (proxy: {w.proxy})" if w.has_proxy() else ""
        print(f"  {w.wallet_id}) {key_preview}{proxy_info}")

    print("  a) Use ALL wallets (parallel mode)")
    print("  q) Quit")
    print()

    while True:
        choice = input("  Enter wallet number, 'a' for all, or 'q': ").strip().lower()

        if choice == 'q':
            return None

        if choice == 'a':
            # TODO: Implement parallel multi-wallet mode
            print("  Multi-wallet mode not yet implemented. Please select one wallet.")
            continue

        try:
            wallet_id = int(choice)
            selected = config.get_wallet(wallet_id)
            if selected:
                return selected
            else:
                print(f"  Wallet #{wallet_id} not found.")
        except ValueError:
            print("  Invalid input.")


def get_order_size_interactive(
    client: OpinionClient,
    config: BotConfig
) -> Optional[Decimal]:
    """
    Let user configure order size.

    Args:
        client: Connected Opinion client
        config: Bot configuration

    Returns:
        Order size in USDT, or None if cancelled
    """
    # Try to get wallet balance
    balance = client.get_balance()

    print("\n" + "-" * 40)
    print("  ORDER SIZE")
    print("-" * 40)

    if balance:
        print(f"  Your balance: ${balance.available:.2f} available")
        print(f"                ${balance.frozen:.2f} in open orders")
        print(f"                ${balance.total:.2f} total")
    else:
        print("  (Could not fetch balance)")

    print()
    print("  How do you want to size your order?")
    print("  1) Fixed USDT amount")
    print("  2) Percentage of available balance")
    print("  q) Cancel")
    print()

    while True:
        choice = input("  Enter 1, 2, or q: ").strip().lower()

        if choice == 'q':
            return None

        if choice == '1':
            # Fixed amount
            default = config.order_size.fixed_amount
            print(f"\n  Enter USDT amount (default: ${default}):")

            amount_str = input(f"  Amount [$]: ").strip()

            if not amount_str:
                return Decimal(str(default))

            try:
                amount = Decimal(amount_str)
                if amount <= 0:
                    print("  Amount must be positive")
                    continue
                return amount
            except:
                print("  Invalid number")
                continue

        elif choice == '2':
            # Percentage of balance
            if not balance or balance.available <= 0:
                print("  Cannot use percentage mode: no available balance")
                continue

            default_pct = config.order_size.percentage
            print(f"\n  Enter percentage of balance (default: {default_pct}%):")

            pct_str = input(f"  Percentage [%]: ").strip()

            if not pct_str:
                pct = Decimal(str(default_pct))
            else:
                try:
                    pct = Decimal(pct_str)
                except:
                    print("  Invalid number")
                    continue

            if pct <= 0 or pct > 100:
                print("  Percentage must be between 0 and 100")
                continue

            amount = balance.available * pct / Decimal("100")
            print(f"  {pct}% of ${balance.available:.2f} = ${amount:.2f}")
            return amount

        else:
            print("  Invalid choice.")


def get_tick_offset_interactive(config: BotConfig) -> Optional[int]:
    """
    Let user configure tick offset (n).

    Args:
        config: Bot configuration

    Returns:
        Tick offset (integer), or None if cancelled
    """
    default = config.trading.default_tick_offset

    print("\n" + "-" * 40)
    print("  TICK OFFSET")
    print("-" * 40)
    print("  How many ticks below the best bid should your order be?")
    print()
    print("  Example: If best bid is 50.0¢ and tick size is 0.1¢:")
    print("    n=1  → your order at 49.9¢")
    print("    n=5  → your order at 49.5¢")
    print("    n=10 → your order at 49.0¢")
    print()
    print(f"  Default: {default}")
    print("  Enter 'q' to cancel")
    print()

    while True:
        n_str = input(f"  Tick offset (n): ").strip().lower()

        if n_str == 'q':
            return None

        if not n_str:
            return default

        try:
            n = int(n_str)
            if n < 1:
                print("  Offset must be at least 1")
                continue
            if n > 50:
                print("  Offset seems very high. Are you sure? (max 50)")
                continue
            return n
        except:
            print("  Invalid number. Enter an integer like 1, 2, or 3.")


def confirm_before_trading(
    client: OpinionClient,
    market: "MarketInfo",
    side: str,
    token_id: str,
    symbol_type: int,
    size: Decimal,
    tick_offset: int,
    tick_size: Decimal
) -> bool:
    """
    Final confirmation before starting the trading loop.

    Args:
        client: Opinion client
        market: Market info
        side: YES or NO
        token_id: Token ID for the chosen side
        symbol_type: 0 for YES, 1 for NO (required for orderbook API)
        size: Order size in USDT
        tick_offset: Tick offset
        tick_size: Tick size for price calculation

    Returns:
        True if user confirms, False otherwise
    """
    from .opinion_client import Orderbook

    # Fetch current orderbook to show calculated price (with correct symbol_type!)
    orderbook = client.get_orderbook(token_id, market.question_id, symbol_type)
    best_bid = orderbook.best_bid() if orderbook else None

    print("\n" + "=" * 60)
    print("  CONFIRM TRADING PARAMETERS")
    print("=" * 60)
    print(f"  Wallet: #{client.wallet_id} ({client.address})")
    print(f"  Market: {market.title}")
    print(f"  Side: {side}")
    print(f"  Order size: ${size:.2f}")
    print(f"  Tick offset: {tick_offset} tick(s) below best bid")

    if best_bid:
        target_price = best_bid - (Decimal(tick_offset) * tick_size)
        target_price = Decimal(str(round(float(target_price), 3)))
        print(f"  Current best bid: {float(best_bid)*100:.1f}¢")
        print(f"  Your initial order price: {float(target_price)*100:.1f}¢")
    else:
        print("  (Could not fetch orderbook for price preview)")

    print("=" * 60)
    print()
    print("  WARNING: This will place REAL orders with REAL money!")
    print()

    confirm = input("  Type 'YES' to start trading: ").strip()

    return confirm.upper() == 'YES'


def run_single_wallet_session(
    client: OpinionClient,
    config: BotConfig
):
    """
    Run a trading session for one wallet.

    This is the main flow:
    1. Select market
    2. Select side
    3. Configure order size
    4. Configure tick offset
    5. Confirm and start pegging loop

    Args:
        client: Connected Opinion client
        config: Bot configuration
    """
    logger = get_logger()
    wallet_id = client.wallet_id

    log_wallet(wallet_id, "Starting trading session")

    # Step 1: Select market
    market = select_market_interactive(client, config.market_filters)
    if not market:
        print("  No market selected. Exiting.")
        return

    # Step 2: Select side
    side = select_side_interactive()
    if not side:
        print("  No side selected. Exiting.")
        return

    # Step 3: Configure order size
    order_size = get_order_size_interactive(client, config)
    if not order_size:
        print("  Order size not configured. Exiting.")
        return

    # Step 4: Configure tick offset
    tick_offset = get_tick_offset_interactive(config)
    if tick_offset is None:
        print("  Tick offset not configured. Exiting.")
        return

    # Get token ID and symbol_type for the selected side
    token_id, symbol_type = get_token_id_for_side(market, side)

    # Step 5: Confirm (with price preview)
    if not confirm_before_trading(
        client, market, side, token_id, symbol_type, order_size, tick_offset, market.tick_size
    ):
        print("  Trading cancelled.")
        return

    # Enable trading (may require on-chain approval)
    print("\n  Enabling trading...")
    if not client.enable_trading():
        print("  Failed to enable trading. Check your wallet has sufficient BNB for gas.")
        return

    state = StrategyState(
        market=market,
        side=side,
        token_id=token_id,
        symbol_type=symbol_type,
        tick_offset=tick_offset,
        order_size_usdt=order_size,
        tick_size=market.tick_size
    )

    # Run the pegging loop
    result = run_pegging_loop(client, state, config.trading)

    # Handle result
    print("\n" + "=" * 60)
    if result == "filled":
        print("  SESSION ENDED: Order was filled")
    elif result == "cancelled":
        print("  SESSION ENDED: Cancelled by user")
    else:
        print(f"  SESSION ENDED: {result}")
    print("=" * 60)


def run_bot(config: BotConfig):
    """
    Main bot entry point.

    Args:
        config: Loaded bot configuration
    """
    logger = get_logger()

    show_welcome_banner()
    print_config_summary(config)

    # Select wallet
    wallet = select_wallet_interactive(config)
    if not wallet:
        print("  No wallet selected. Exiting.")
        return

    # Create and connect client
    client = OpinionClient(wallet, config.network)

    print(f"\n  Connecting wallet #{wallet.wallet_id}...")

    if not client.connect():
        print("  Failed to connect. Check your private key and network settings.")
        return

    print(f"  Connected! Address: {client.address}")

    try:
        # Main session loop
        while True:
            run_single_wallet_session(client, config)

            print("\n" + "-" * 40)
            print("  What would you like to do?")
            print("  1) Start a new trading session")
            print("  2) Liquidate a position (sell at break-even)")
            print("  3) Switch wallet")
            print("  q) Quit")
            print()

            choice = input("  Enter 1, 2, 3, or q: ").strip().lower()

            if choice == 'q':
                break
            elif choice == '3':
                client.disconnect()
                wallet = select_wallet_interactive(config)
                if not wallet:
                    break
                client = OpinionClient(wallet, config.network)
                if not client.connect():
                    print("  Failed to connect new wallet.")
                    break
            elif choice == '2':
                # Run liquidator
                run_liquidator(client)
            elif choice == '1':
                continue
            else:
                print("  Invalid choice. Starting new session...")

    finally:
        print("\n  Cleaning up...")
        client.disconnect()
        print("  Goodbye!")
