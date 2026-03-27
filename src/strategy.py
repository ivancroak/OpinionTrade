"""
Trading strategy module.

This module implements the "pegged limit order" strategy:
- Place limit buy orders below the best bid
- Never be the best bid (always stay behind other traders)
- Automatically adjust when the orderbook changes
- Handle fills with user prompts
"""

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple, List

from .opinion_client import OpinionClient, MarketInfo, Orderbook, OrderInfo
from .config_loader import TradingConfig
from .utils.logging_utils import get_logger, log_trade, log_wallet


@dataclass
class StrategyState:
    """
    Current state of the strategy for one market/side.

    Tracks:
    - Current order (if any)
    - Fill status
    - Target price calculations
    """
    market: MarketInfo
    side: str  # "YES" or "NO"
    token_id: str
    symbol_type: int  # 0 for YES, 1 for NO (required for orderbook API)
    tick_offset: int  # n ticks below best bid
    order_size_usdt: Decimal
    tick_size: Decimal = Decimal("0.001")  # Default 0.1 cent (Opinion.trade uses 0.001 increments)

    # Order tracking
    current_order_id: Optional[str] = None
    current_order_price: Optional[Decimal] = None
    current_order_shares: Decimal = Decimal("0")  # Track shares for fill detection
    last_external_best_bid: Optional[Decimal] = None

    # Fill tracking
    total_filled: Decimal = Decimal("0")
    total_spent: Decimal = Decimal("0")

    # Auto-flip mode: automatically sell filled shares at best ask and continue
    auto_flip: bool = True  # Default to True for liquid farming mode

    # Track pending sell orders for auto-flip
    pending_sell_order_id: Optional[str] = None
    pending_sell_shares: Decimal = Decimal("0")

    @property
    def has_active_order(self) -> bool:
        return self.current_order_id is not None

    @property
    def has_pending_sell(self) -> bool:
        return self.pending_sell_order_id is not None


def get_token_id_for_side(market: MarketInfo, side: str) -> Tuple[str, int]:
    """
    Get the token ID and symbol_type for YES or NO side.

    Args:
        market: Market info
        side: "YES" or "NO"

    Returns:
        Tuple of (token_id, symbol_type)
        symbol_type: 0 for YES, 1 for NO (required by orderbook API)
    """
    if side.upper() == "YES":
        return market.yes_token_id, 0
    else:
        return market.no_token_id, 1


def compute_external_best_bid(
    orderbook: Orderbook,
    my_order_price: Optional[Decimal] = None,
    my_order_size: Optional[Decimal] = None
) -> Optional[Decimal]:
    """
    Compute the best bid from OTHER traders (excluding my order).

    Simple approach: if my order price matches the best bid, use the second-best bid.
    More accurate approach would use order IDs, but that requires API support.

    Args:
        orderbook: Current orderbook
        my_order_price: My current order's price (if any)
        my_order_size: My current order's size (if any)

    Returns:
        External best bid price, or None if no bids
    """
    if not orderbook.bids:
        return None

    # If I don't have an order, best bid is the external best bid
    if my_order_price is None:
        return orderbook.bids[0].price

    # If my order is at best bid, we need to check if there's other volume
    best = orderbook.bids[0]

    if best.price == my_order_price:
        # Check if there's more volume than just my order
        if my_order_size and best.amount > my_order_size:
            # There's other volume at this price level
            return best.price
        elif len(orderbook.bids) > 1:
            # Use second-best bid as external best
            return orderbook.bids[1].price
        else:
            # I'm the only bid - no external best
            return None
    else:
        # My order is not at best bid, so best bid is external
        return best.price


def compute_target_price(
    external_best_bid: Decimal,
    tick_offset: int,
    tick_size: Decimal,
    min_price: Decimal = Decimal("0.001")
) -> Decimal:
    """
    Compute the target price for our order.

    Formula: target = external_best_bid - (tick_offset * tick_size)

    Args:
        external_best_bid: Best bid from other traders
        tick_offset: Number of ticks below best bid (n)
        tick_size: Price increment (e.g., 0.001)
        min_price: Minimum allowed price

    Returns:
        Target price, clamped to min_price if needed, truncated to 3 decimals
    """
    target = external_best_bid - (Decimal(tick_offset) * tick_size)

    # Don't go below minimum price
    if target < min_price:
        target = min_price

    # Truncate to 3 decimal places (matches reference bot's round_cut function)
    # Using truncation instead of rounding to avoid floating-point issues
    target_float = float(target)
    target_truncated = int(target_float * 1000) / 1000
    target = Decimal(str(target_truncated))

    return target


def check_order_fill(
    client: OpinionClient,
    order_id: str,
    expected_shares: Decimal = Decimal("0")
) -> Tuple[bool, Decimal, Decimal]:
    """
    Check if an order has been filled (partially or fully).

    Args:
        client: Opinion client
        order_id: Order ID to check
        expected_shares: Expected share size (used when order is removed after fill)

    Returns:
        Tuple of (has_fill, filled_amount, remaining_amount)
    """
    logger = get_logger()
    order = client.get_order(order_id)

    if not order:
        # Order not found - it was fully filled and removed from active orders!
        # Use expected_shares as the filled amount
        if expected_shares > 0:
            logger.info(f"Order {order_id} not found in active orders - assuming fully filled ({expected_shares} shares)")
            return True, expected_shares, Decimal("0")
        else:
            # Fallback: can't determine filled amount
            logger.warning(f"Order {order_id} not found and no expected_shares provided")
            return True, Decimal("0"), Decimal("0")

    has_fill = order.filled_size > 0
    return has_fill, order.filled_size, order.remaining_size


def handle_fill_menu(
    client: OpinionClient,
    state: StrategyState,
    filled_size: Decimal,
    fill_price: Decimal
) -> str:
    """
    Display fill menu and get user's choice.

    Args:
        client: Opinion client
        state: Current strategy state
        filled_size: Amount that was filled
        fill_price: Average fill price

    Returns:
        User's choice: "close", "limit", or "nothing"
    """
    logger = get_logger()

    print("\n" + "=" * 60)
    print("  ORDER FILLED!")
    print("=" * 60)
    print(f"  Market: {state.market.title}")
    print(f"  Side: {state.side}")
    print(f"  Filled: {filled_size} shares @ {fill_price}")
    print(f"  Total spent: ~${float(filled_size) * float(fill_price):.2f}")
    print("=" * 60)
    print()
    print("  What would you like to do with this position?")
    print()
    print("  1) CLOSE - Sell position immediately (market order)")
    print("  2) LIMIT - Place a limit sell order (you specify price)")
    print("  3) HOLD  - Keep position, stop trading this market")
    print()

    while True:
        choice = input("  Enter 1, 2, or 3: ").strip()

        if choice == '1':
            return "close"
        elif choice == '2':
            return "limit"
        elif choice == '3':
            return "nothing"
        else:
            print("  Invalid choice. Please enter 1, 2, or 3.")


def execute_close_position(
    client: OpinionClient,
    state: StrategyState,
    shares: Decimal
) -> bool:
    """
    Close a position using an aggressive limit order (crossing spread).

    Args:
        client: Opinion client
        state: Strategy state
        shares: Number of shares to sell

    Returns:
        True if close order placed successfully
    """
    logger = get_logger()

    # Get current orderbook to find best ask (with correct symbol_type!)
    orderbook = client.get_orderbook(state.token_id, state.market.question_id, state.symbol_type)

    if not orderbook or not orderbook.asks:
        logger.error("Cannot close: no asks in orderbook")
        return False

    # Use best ask price to ensure immediate fill
    close_price = orderbook.asks[0].price

    log_trade("CLOSING", f"Selling {shares} shares @ {close_price}")

    order_id = client.place_limit_order(
        market_id=state.market.market_id,
        token_id=state.token_id,
        side="SELL",
        price=close_price,
        size_usdt=shares * close_price  # Approximate
    )

    return order_id is not None


def execute_limit_sell(
    client: OpinionClient,
    state: StrategyState,
    shares: Decimal
) -> bool:
    """
    Place a limit sell order at user-specified price.

    Args:
        client: Opinion client
        state: Strategy state
        shares: Number of shares to sell

    Returns:
        True if order placed successfully
    """
    logger = get_logger()

    # Get current market price for reference (with correct symbol_type!)
    orderbook = client.get_orderbook(state.token_id, state.market.question_id, state.symbol_type)
    current_bid = orderbook.best_bid() if orderbook else None
    current_ask = orderbook.best_ask() if orderbook else None

    print()
    if current_bid and current_ask:
        print(f"  Current market: Bid {current_bid} / Ask {current_ask}")
    print(f"  You have {shares} shares to sell.")
    print()

    while True:
        price_input = input("  Enter your limit sell price (e.g., 0.65): ").strip()

        try:
            sell_price = Decimal(price_input)
            if sell_price <= 0 or sell_price > 1:
                print("  Price must be between 0 and 1")
                continue

            break
        except:
            print("  Invalid price. Enter a number like 0.65")

    log_trade("LIMIT SELL", f"Placing sell order: {shares} shares @ {sell_price}")

    order_id = client.place_limit_order(
        market_id=state.market.market_id,
        token_id=state.token_id,
        side="SELL",
        price=sell_price,
        size_usdt=shares * sell_price
    )

    if order_id:
        print(f"  Limit sell order placed successfully!")
        return True
    else:
        print("  Failed to place sell order.")
        return False


def execute_auto_flip(
    client: OpinionClient,
    state: StrategyState,
    shares: Decimal
) -> Optional[str]:
    """
    Automatically flip position by selling at best ask price.

    This is used in liquid farming mode - when we accidentally get filled,
    immediately place a sell order at the current best ask to exit the position.

    Args:
        client: Opinion client
        state: Strategy state
        shares: Number of shares to sell

    Returns:
        Order ID if sell order placed successfully, None otherwise
    """
    logger = get_logger()
    wallet_id = client.wallet_id

    # Get current orderbook to find best ask (with correct symbol_type!)
    orderbook = client.get_orderbook(state.token_id, state.market.question_id, state.symbol_type)

    if not orderbook or not orderbook.asks:
        logger.error("AUTO-FLIP: Cannot sell - no asks in orderbook")
        return None

    # Use best ask price for immediate execution
    best_ask = orderbook.asks[0].price

    log_trade("AUTO-FLIP", f"Selling {shares} shares @ {best_ask} (best ask)")
    log_wallet(wallet_id, f"Auto-flip: selling {shares} shares @ {best_ask}")

    order_id = client.place_limit_order(
        market_id=state.market.market_id,
        token_id=state.token_id,
        side="SELL",
        price=best_ask,
        size_usdt=shares * best_ask
    )

    if order_id:
        log_trade("AUTO-FLIP", f"Sell order placed: {order_id}")
        return order_id
    else:
        logger.error("AUTO-FLIP: Failed to place sell order")
        return None


def check_sell_order_filled(
    client: OpinionClient,
    order_id: str
) -> Tuple[bool, Decimal, Decimal]:
    """
    Check if a sell order has been filled.

    Args:
        client: Opinion client
        order_id: Sell order ID to check

    Returns:
        Tuple of (is_complete, filled_amount, remaining_amount)
        is_complete is True if order is fully filled or cancelled
    """
    order = client.get_order(order_id)

    if not order:
        # Order not found - probably fully filled and removed
        return True, Decimal("0"), Decimal("0")

    is_complete = order.remaining_size == 0 or order.status in ["FILLED", "CANCELLED"]
    return is_complete, order.filled_size, order.remaining_size


def run_pegging_loop(
    client: OpinionClient,
    state: StrategyState,
    config: TradingConfig
) -> str:
    """
    Main pegging loop - keeps order positioned below best bid.

    This loop:
    1. Fetches orderbook periodically
    2. Computes target price
    3. Places/adjusts orders
    4. Detects fills and handles them

    Args:
        client: Opinion client
        state: Strategy state (will be modified)
        config: Trading configuration

    Returns:
        Exit reason: "filled", "cancelled", "error"
    """
    logger = get_logger()
    wallet_id = client.wallet_id

    log_wallet(wallet_id, f"Starting pegging loop for {state.market.title} ({state.side})")
    log_wallet(wallet_id, f"Tick offset: {state.tick_offset}, Size: ${state.order_size_usdt}")

    print("\n" + "-" * 60)
    print("  PEGGING LOOP STARTED")
    print("-" * 60)
    print(f"  Market: {state.market.title}")
    print(f"  Side: {state.side}")
    print(f"  Strategy: Place bid {state.tick_offset} tick(s) below best external bid")
    print(f"  Order size: ${state.order_size_usdt}")
    print(f"  Poll interval: {config.poll_interval_seconds}s")
    if state.auto_flip:
        print(f"  Auto-flip: ENABLED (auto-sell at best ask if filled)")
    else:
        print(f"  Auto-flip: DISABLED (manual handling on fill)")
    print()
    print("  Press Ctrl+C to stop")
    print("-" * 60)

    iteration = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5  # Stop after 5 consecutive failures

    try:
        while True:
            iteration += 1

            # Step 1: Fetch orderbook (with correct symbol_type for YES/NO side!)
            orderbook = client.get_orderbook(state.token_id, state.market.question_id, state.symbol_type)

            if not orderbook:
                consecutive_failures += 1
                wait_time = min(config.poll_interval_seconds * consecutive_failures, 30)  # Max 30s backoff
                logger.warning(f"Failed to fetch orderbook ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}), retrying in {wait_time}s...")

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Too many consecutive failures. Stopping pegging loop.")
                    logger.error("Please check your internet connection and restart the bot.")
                    # Cancel any active order before stopping
                    if state.has_active_order:
                        try:
                            client.cancel_order(state.current_order_id)
                            log_wallet(wallet_id, "Cancelled active order before stopping")
                        except:
                            pass
                    return "error"

                time.sleep(wait_time)
                continue

            # Reset failure counter on success
            consecutive_failures = 0

            # Step 1.5: Check pending sell orders from auto-flip
            if state.has_pending_sell:
                try:
                    is_complete, sold, remaining = check_sell_order_filled(
                        client, state.pending_sell_order_id
                    )

                    if is_complete:
                        if sold > 0:
                            log_trade("AUTO-FLIP COMPLETE", f"Sold {sold} shares")
                            log_wallet(wallet_id, f"AUTO-FLIP: Sold {sold} shares successfully")
                            print(f"  [AUTO-FLIP] Sell complete: {sold} shares sold")
                        else:
                            log_wallet(wallet_id, "AUTO-FLIP: Sell order completed (might have been cancelled)")

                        # Clear pending sell state
                        state.pending_sell_order_id = None
                        state.pending_sell_shares = Decimal("0")
                    else:
                        # Still pending - maybe re-peg if price moved?
                        # For now just wait, the order is at best ask so should fill quickly
                        pass
                except Exception as e:
                    logger.warning(f"Error checking sell order: {e}")

            # Step 2: Compute external best bid
            external_best = compute_external_best_bid(
                orderbook,
                my_order_price=state.current_order_price,
                my_order_size=state.order_size_usdt
            )

            if external_best is None:
                logger.warning("No external bids found in orderbook")
                time.sleep(config.poll_interval_seconds)
                continue

            state.last_external_best_bid = external_best

            # Step 3: Compute target price
            target_price = compute_target_price(
                external_best,
                state.tick_offset,
                state.tick_size
            )

            # Step 4: Check fills if we have an active order
            if state.has_active_order:
                try:
                    fill_result = check_order_fill(
                        client, state.current_order_id,
                        expected_shares=state.current_order_shares  # Pass expected shares for fill detection
                    )
                    if fill_result is None:
                        logger.warning("Could not check order fill status, skipping...")
                        time.sleep(config.poll_interval_seconds)
                        continue
                    has_fill, filled, remaining = fill_result
                except Exception as e:
                    logger.warning(f"Error checking order fill: {e}, skipping...")
                    time.sleep(config.poll_interval_seconds)
                    continue

                if has_fill and filled > 0:
                    log_trade("FILL DETECTED", f"Filled: {filled}, Remaining: {remaining}")

                    # Cancel remaining if partial fill
                    if remaining > 0:
                        client.cancel_order(state.current_order_id)

                    state.total_filled += filled
                    state.current_order_id = None
                    state.current_order_price = None
                    state.current_order_shares = Decimal("0")  # Clear tracked shares

                    # AUTO-FLIP MODE: Automatically sell at best ask and continue
                    if state.auto_flip:
                        log_wallet(wallet_id, f"AUTO-FLIP: Got filled {filled} shares, selling at best ask...")
                        print(f"\n  [AUTO-FLIP] Bought {filled} shares - immediately selling at best ask")

                        sell_order_id = execute_auto_flip(client, state, filled)

                        if sell_order_id:
                            state.pending_sell_order_id = sell_order_id
                            state.pending_sell_shares = filled
                            log_wallet(wallet_id, f"AUTO-FLIP: Sell order placed, continuing pegging loop")
                            print(f"  [AUTO-FLIP] Sell order placed: {sell_order_id}")
                        else:
                            logger.error("AUTO-FLIP: Failed to place sell order - stopping for manual intervention")
                            return "error"

                        # Continue the loop - don't exit
                        # The sell order will execute at market, we continue placing buy orders
                    else:
                        # MANUAL MODE: Show fill menu and wait for user decision
                        choice = handle_fill_menu(
                            client, state, filled, target_price
                        )

                        if choice == "close":
                            execute_close_position(client, state, filled)
                        elif choice == "limit":
                            execute_limit_sell(client, state, filled)

                        return "filled"

            # Step 5: Check if we need to adjust our order
            if state.has_active_order:
                # Check if our price matches target
                price_changed = state.current_order_price != target_price

                # Check if we've become best bid (bad!)
                is_best_bid = (
                    orderbook and orderbook.bids and
                    orderbook.bids[0].price == state.current_order_price
                )

                if price_changed or is_best_bid:
                    reason = "price drift" if price_changed else "became best bid"
                    log_wallet(wallet_id, f"Adjusting order ({reason})")

                    # CRITICAL: Check for partial fills BEFORE cancelling!
                    # This prevents losing track of filled shares
                    try:
                        pre_cancel_check = check_order_fill(
                            client, state.current_order_id,
                            expected_shares=state.current_order_shares
                        )
                        if pre_cancel_check:
                            _, filled_before_cancel, remaining = pre_cancel_check

                            if filled_before_cancel > 0:
                                # Order was partially filled while sitting on the book!
                                log_trade("PARTIAL FILL DETECTED", f"Filled {filled_before_cancel} shares before cancel")
                                log_wallet(wallet_id, f"Partial fill detected: {filled_before_cancel} shares")

                                # Cancel the remaining portion
                                try:
                                    client.cancel_order(state.current_order_id)
                                except Exception as e:
                                    logger.warning(f"Error cancelling partially filled order: {e}")

                                state.total_filled += filled_before_cancel
                                state.current_order_id = None
                                state.current_order_price = None
                                state.current_order_shares = Decimal("0")

                                # Handle the filled shares
                                if state.auto_flip:
                                    log_wallet(wallet_id, f"AUTO-FLIP: Partial fill of {filled_before_cancel} shares, selling at best ask...")
                                    print(f"\n  [AUTO-FLIP] Partial fill: {filled_before_cancel} shares - immediately selling at best ask")

                                    sell_order_id = execute_auto_flip(client, state, filled_before_cancel)

                                    if sell_order_id:
                                        state.pending_sell_order_id = sell_order_id
                                        state.pending_sell_shares = filled_before_cancel
                                        log_wallet(wallet_id, f"AUTO-FLIP: Sell order placed for partial fill")
                                        print(f"  [AUTO-FLIP] Sell order placed: {sell_order_id}")
                                    else:
                                        logger.error("AUTO-FLIP: Failed to place sell order for partial fill - stopping")
                                        return "error"
                                else:
                                    # Manual mode - show fill menu
                                    choice = handle_fill_menu(
                                        client, state, filled_before_cancel, state.current_order_price
                                    )

                                    if choice == "close":
                                        execute_close_position(client, state, filled_before_cancel)
                                    elif choice == "limit":
                                        execute_limit_sell(client, state, filled_before_cancel)

                                    return "filled"

                                # Continue to place new order
                                continue
                    except Exception as e:
                        logger.warning(f"Error checking for partial fill before cancel: {e}")

                    # No partial fill detected, safe to cancel
                    try:
                        client.cancel_order(state.current_order_id)
                    except Exception as e:
                        logger.warning(f"Error cancelling order: {e}")
                    state.current_order_id = None
                    state.current_order_price = None
                    state.current_order_shares = Decimal("0")  # Clear tracked shares

            # Step 6: Place order if we don't have one
            if not state.has_active_order:
                # Calculate expected shares for fill detection
                expected_shares = state.order_size_usdt / target_price

                log_wallet(wallet_id, f"Placing order @ {target_price} (ext best: {external_best}), ~{expected_shares:.2f} shares")

                try:
                    order_id = client.place_limit_order(
                        market_id=state.market.market_id,
                        token_id=state.token_id,
                        side="BUY",
                        price=target_price,
                        size_usdt=state.order_size_usdt
                    )

                    if order_id:
                        state.current_order_id = order_id
                        state.current_order_price = target_price
                        state.current_order_shares = expected_shares  # Track shares for fill detection
                    else:
                        logger.error("Failed to place order")
                except Exception as e:
                    logger.error(f"Error placing order: {e}")

            # Log status periodically
            if iteration % 10 == 0:
                logger.debug(
                    f"Loop #{iteration}: ext_best={external_best}, "
                    f"target={target_price}, order={state.current_order_id is not None}"
                )

            # Wait before next iteration
            time.sleep(config.poll_interval_seconds)

    except KeyboardInterrupt:
        print("\n\n  Stopping pegging loop (Ctrl+C pressed)...")

        # Cancel any active buy order
        if state.has_active_order:
            log_wallet(wallet_id, "Cancelling active buy order...")
            client.cancel_order(state.current_order_id)

        # Note: We do NOT cancel pending sell orders on Ctrl+C
        # The user might still want those to execute
        if state.has_pending_sell:
            print(f"  Note: You have a pending sell order ({state.pending_sell_order_id})")
            print(f"        for {state.pending_sell_shares} shares - it will continue to execute")

        return "cancelled"

    except Exception as e:
        logger.error(f"Error in pegging loop: {e}")

        # Try to cancel active buy order
        if state.has_active_order:
            try:
                client.cancel_order(state.current_order_id)
            except:
                pass

        # Don't cancel sell orders on error - let them try to execute

        return "error"
