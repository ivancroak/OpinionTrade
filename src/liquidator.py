"""
Position Liquidation Module.

This module provides functionality to liquidate positions at break-even or profit:
- Monitors orderbook for bids at or above average cost
- Calculates available liquidity at profitable prices
- Places sell orders when favorable conditions are met
- Supports partial fills to maximize exit efficiency
"""

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple

from .opinion_client import OpinionClient, PositionInfo, Orderbook, MarketInfo
from .utils.logging_utils import get_logger, log_wallet, log_trade


@dataclass
class LiquidationTarget:
    """A position targeted for liquidation."""
    position: PositionInfo
    market: Optional[MarketInfo]
    min_price: Decimal  # Minimum acceptable price (avg_cost)
    shares_to_sell: Decimal
    question_id: str = ""
    symbol_type: int = 0  # 0 for YES, 1 for NO


@dataclass
class LiquidationOpportunity:
    """An opportunity to liquidate shares at a favorable price."""
    target: LiquidationTarget
    available_shares: Decimal  # Shares that can be sold at >= min_price
    available_value: Decimal  # USD value available
    best_bid: Decimal
    bid_levels_used: int  # Number of bid levels needed


def calculate_liquidation_opportunity(
    orderbook: Orderbook,
    target: LiquidationTarget
) -> Optional[LiquidationOpportunity]:
    """
    Calculate how many shares can be sold at or above the minimum price.

    This function walks through the bid side of the orderbook and calculates
    how many shares can be sold at prices >= min_price.

    Args:
        orderbook: Current orderbook
        target: Liquidation target with min_price and shares to sell

    Returns:
        LiquidationOpportunity if any shares can be sold, None otherwise
    """
    if not orderbook or not orderbook.bids:
        return None

    available_shares = Decimal("0")
    available_value = Decimal("0")
    levels_used = 0

    for bid in orderbook.bids:
        # Stop if bid is below minimum price
        if bid.price < target.min_price:
            break

        levels_used += 1

        # Calculate how many shares we can sell at this level
        remaining_to_sell = target.shares_to_sell - available_shares
        shares_at_level = min(bid.amount, remaining_to_sell)

        available_shares += shares_at_level
        available_value += shares_at_level * bid.price

        # Stop if we have enough liquidity
        if available_shares >= target.shares_to_sell:
            break

    if available_shares <= 0:
        return None

    return LiquidationOpportunity(
        target=target,
        available_shares=available_shares,
        available_value=available_value,
        best_bid=orderbook.bids[0].price,
        bid_levels_used=levels_used
    )


def execute_liquidation(
    client: OpinionClient,
    opportunity: LiquidationOpportunity,
    use_market_order: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Execute a liquidation by placing a sell order.

    Args:
        client: Opinion client
        opportunity: The liquidation opportunity to execute
        use_market_order: If True, use market order; otherwise use aggressive limit

    Returns:
        Tuple of (success, order_id)
    """
    logger = get_logger()
    target = opportunity.target

    # Determine sell price - use best bid for aggressive limit order
    # This ensures immediate execution while still getting a good price
    sell_price = opportunity.best_bid
    shares_to_sell = opportunity.available_shares

    log_trade(
        "LIQUIDATION",
        f"Selling {shares_to_sell:.2f} shares @ {float(sell_price)*100:.2f}¢ "
        f"(min: {float(target.min_price)*100:.2f}¢)"
    )

    if not target.market:
        logger.error("Cannot liquidate: market info not available")
        return False, None

    # Calculate order value
    order_value = shares_to_sell * sell_price

    # Place the sell order
    order_id = client.place_limit_order(
        market_id=target.market.market_id,
        token_id=target.position.token_id,
        side="SELL",
        price=sell_price,
        size_usdt=order_value
    )

    if order_id:
        log_wallet(client.wallet_id, f"Liquidation order placed: {order_id[:16]}...")
        return True, order_id
    else:
        logger.error("Failed to place liquidation order")
        return False, None


def run_liquidation_loop(
    client: OpinionClient,
    target: LiquidationTarget,
    poll_interval: float = 2.0,
    max_wait_time: float = 3600.0  # 1 hour max wait
) -> str:
    """
    Main liquidation loop - monitors orderbook and liquidates when profitable.

    This loop:
    1. Fetches the orderbook periodically
    2. Calculates if liquidation is possible at break-even or better
    3. If yes, executes the liquidation
    4. Monitors the sell order until filled

    Args:
        client: Opinion client
        target: Liquidation target
        poll_interval: Seconds between orderbook checks
        max_wait_time: Maximum time to wait for opportunity

    Returns:
        Exit reason: "liquidated", "cancelled", "timeout", "error"
    """
    logger = get_logger()
    wallet_id = client.wallet_id

    log_wallet(wallet_id, f"Starting liquidation monitor")
    log_wallet(wallet_id, f"Position: {target.shares_to_sell:.2f} shares @ avg {float(target.min_price)*100:.2f}¢")
    log_wallet(wallet_id, f"Min sell price: {float(target.min_price)*100:.2f}¢")

    print("\n" + "-" * 60)
    print("  LIQUIDATION MONITOR STARTED")
    print("-" * 60)
    print(f"  Market: {target.position.market_title}")
    print(f"  Side: {target.position.side}")
    print(f"  Shares to sell: {target.shares_to_sell:.2f}")
    print(f"  Average cost: {float(target.min_price)*100:.2f}¢")
    print(f"  Break-even value: ${float(target.shares_to_sell * target.min_price):.2f}")
    print()
    print("  Waiting for bids at or above average cost...")
    print("  Press Ctrl+C to stop")
    print("-" * 60)

    start_time = time.time()
    pending_order_id: Optional[str] = None
    remaining_shares = target.shares_to_sell

    try:
        while remaining_shares > 0:
            elapsed = time.time() - start_time

            if elapsed > max_wait_time:
                logger.warning("Max wait time exceeded")
                return "timeout"

            # Check pending order status
            if pending_order_id:
                order = client.get_order(pending_order_id)

                if not order:
                    # Order not found - probably fully filled
                    log_trade("LIQUIDATION COMPLETE", f"Order {pending_order_id[:16]}... filled")
                    print(f"\n  [LIQUIDATED] Order filled completely!")
                    return "liquidated"

                if order.filled_size > 0:
                    filled = order.filled_size
                    remaining_shares -= filled
                    log_wallet(wallet_id, f"Partial liquidation: {filled:.2f} shares sold")
                    print(f"  [PARTIAL] Sold {filled:.2f} shares, {remaining_shares:.2f} remaining")

                    if remaining_shares <= 0:
                        return "liquidated"

                    # Update target for remaining shares
                    target.shares_to_sell = remaining_shares

                    # Cancel remaining order if price moved
                    if order.remaining_size > 0:
                        client.cancel_order(pending_order_id)
                        pending_order_id = None

                continue

            # Fetch orderbook
            orderbook = client.get_orderbook(
                target.position.token_id,
                target.question_id,
                target.symbol_type
            )

            if not orderbook:
                logger.warning("Failed to fetch orderbook")
                time.sleep(poll_interval)
                continue

            # Check for liquidation opportunity
            opportunity = calculate_liquidation_opportunity(orderbook, target)

            if opportunity and opportunity.available_shares > 0:
                # We have an opportunity!
                pct_available = (opportunity.available_shares / target.shares_to_sell) * 100

                print(f"\n  [OPPORTUNITY] {opportunity.available_shares:.2f} shares "
                      f"({pct_available:.1f}%) available @ >= {float(target.min_price)*100:.2f}¢")
                print(f"  [OPPORTUNITY] Best bid: {float(opportunity.best_bid)*100:.2f}¢, "
                      f"using {opportunity.bid_levels_used} level(s)")

                # Execute liquidation
                success, order_id = execute_liquidation(client, opportunity)

                if success and order_id:
                    pending_order_id = order_id
                    print(f"  [EXECUTING] Sell order placed: {order_id[:16]}...")
                else:
                    logger.error("Failed to execute liquidation")
                    time.sleep(poll_interval * 2)  # Wait longer on failure

            else:
                # No opportunity yet
                if orderbook.bids:
                    best_bid = orderbook.bids[0].price
                    gap = target.min_price - best_bid
                    gap_pct = (gap / target.min_price) * 100

                    # Only print status every 30 seconds to reduce noise
                    if int(elapsed) % 30 == 0:
                        print(f"  [WAITING] Best bid: {float(best_bid)*100:.2f}¢ "
                              f"(need {float(target.min_price)*100:.2f}¢, gap: {float(gap_pct):.2f}%)")
                else:
                    if int(elapsed) % 30 == 0:
                        print("  [WAITING] No bids in orderbook")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\n  Stopping liquidation monitor (Ctrl+C pressed)...")

        if pending_order_id:
            print(f"  Note: You have a pending sell order ({pending_order_id[:16]}...)")
            cancel_choice = input("  Cancel pending order? (y/n): ").strip().lower()
            if cancel_choice == 'y':
                client.cancel_order(pending_order_id)
                print("  Order cancelled.")
            else:
                print("  Order left open.")

        return "cancelled"

    except Exception as e:
        logger.error(f"Error in liquidation loop: {e}")
        return "error"

    return "liquidated"


def select_position_interactive(client: OpinionClient) -> Optional[LiquidationTarget]:
    """
    Let user select a position to liquidate.

    Args:
        client: Connected Opinion client

    Returns:
        LiquidationTarget if selected, None if cancelled
    """
    logger = get_logger()

    print("\n" + "-" * 60)
    print("  SELECT POSITION TO LIQUIDATE")
    print("-" * 60)
    print("  Fetching your positions...")

    all_positions = client.get_positions()

    # Filter out positions with 0 shares
    positions = [p for p in all_positions if p.shares > 0]

    if not positions:
        print("  No open positions found (with shares > 0).")
        return None

    print(f"\n  Found {len(positions)} position(s) with shares:\n")

    for i, pos in enumerate(positions, 1):
        value = pos.shares * pos.avg_price
        pnl = pos.current_value - value
        pnl_pct = (pnl / value * 100) if value > 0 else 0

        print(f"  {i}) {pos.market_title}")
        print(f"     Side: {pos.side} - {pos.shares:.2f} shares")
        print(f"     Avg cost: {float(pos.avg_price)*100:.2f}¢")
        print(f"     Cost basis: ${float(value):.2f}")
        print(f"     Current value: ${float(pos.current_value):.2f}")
        print(f"     P&L: ${float(pnl):.2f} ({pnl_pct:+.1f}%)")
        print()

    print("  q) Cancel")
    print()

    while True:
        choice = input("  Select position number: ").strip().lower()

        if choice == 'q':
            return None

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(positions):
                selected = positions[idx]

                # Check for valid avg_price
                if selected.avg_price <= 0:
                    print("  Warning: This position has no recorded average price.")
                    print("  Cannot determine break-even price.")
                    continue

                # Fetch market info for this position
                market = client.get_market(selected.market_id)

                # Determine symbol_type based on side
                symbol_type = 0 if selected.side == "YES" else 1

                # Get question_id from market if available
                question_id = market.question_id if market else ""

                return LiquidationTarget(
                    position=selected,
                    market=market,
                    min_price=selected.avg_price,
                    shares_to_sell=selected.shares,
                    question_id=question_id,
                    symbol_type=symbol_type
                )
            else:
                print("  Invalid selection.")
        except ValueError:
            print("  Please enter a number.")


def run_liquidator(client: OpinionClient):
    """
    Main entry point for the liquidation mode.

    Args:
        client: Connected Opinion client
    """
    logger = get_logger()

    print("\n" + "=" * 60)
    print("   POSITION LIQUIDATOR")
    print("   Sell at Break-Even or Better")
    print("=" * 60)

    # Select position
    target = select_position_interactive(client)

    if not target:
        print("  No position selected. Exiting.")
        return

    # Confirm
    print("\n" + "=" * 60)
    print("  CONFIRM LIQUIDATION PARAMETERS")
    print("=" * 60)
    print(f"  Market: {target.position.market_title}")
    print(f"  Position: {target.position.side}")
    print(f"  Shares: {target.shares_to_sell:.2f}")
    print(f"  Min sell price: {float(target.min_price)*100:.2f}¢ (your avg cost)")
    print(f"  Break-even value: ${float(target.shares_to_sell * target.min_price):.2f}")
    print()
    print("  The bot will monitor the orderbook and sell when bids")
    print("  are available at or above your average cost.")
    print("=" * 60)
    print()

    confirm = input("  Type 'YES' to start liquidation monitor: ").strip()

    if confirm.upper() != 'YES':
        print("  Liquidation cancelled.")
        return

    # Run liquidation loop
    result = run_liquidation_loop(client, target)

    # Handle result
    print("\n" + "=" * 60)
    if result == "liquidated":
        print("  LIQUIDATION COMPLETE!")
        print("  Position sold at break-even or better.")
    elif result == "cancelled":
        print("  LIQUIDATION CANCELLED")
    elif result == "timeout":
        print("  LIQUIDATION TIMEOUT")
        print("  No favorable opportunity found within time limit.")
    else:
        print(f"  LIQUIDATION ENDED: {result}")
    print("=" * 60)
