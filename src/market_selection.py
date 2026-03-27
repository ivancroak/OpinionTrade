"""
Market selection module.

This module handles:
- Fetching and filtering markets
- Ranking markets by volume or open interest
- Displaying market lists to the user
"""

from dataclasses import dataclass
from typing import List, Optional

from .opinion_client import OpinionClient, MarketInfo
from .config_loader import MarketFilters
from .utils.logging_utils import get_logger


@dataclass
class RankedMarket:
    """A market with its ranking info."""
    rank: int
    market: MarketInfo
    rank_value: float  # The value used for ranking (volume or OI)


def fetch_active_markets(client: OpinionClient, limit: int = 200) -> List[MarketInfo]:
    """
    Fetch all active (tradeable) markets.

    Fetches from both Binary and Categorical market types to ensure all markets are included.

    Args:
        client: Opinion client instance
        limit: Maximum markets to fetch per market type (default 200)

    Returns:
        List of active markets
    """
    return client.get_markets(status="activated", limit=limit)


def filter_markets(
    markets: List[MarketInfo],
    filters: MarketFilters
) -> List[MarketInfo]:
    """
    Filter markets based on minimum volume/OI thresholds.

    Args:
        markets: List of markets to filter
        filters: Filter criteria

    Returns:
        Filtered list of markets
    """
    filtered = []

    for m in markets:
        # Check minimum 24h volume
        if m.volume_24h < filters.min_24h_volume:
            continue

        # Add other filters here as needed
        filtered.append(m)

    return filtered


def rank_by_volume(
    markets: List[MarketInfo],
    top_n: int = 10
) -> List[RankedMarket]:
    """
    Rank markets by 24h volume (highest first).

    Args:
        markets: List of markets
        top_n: Number of top markets to return

    Returns:
        List of ranked markets
    """
    # Sort by volume descending
    sorted_markets = sorted(markets, key=lambda m: m.volume_24h, reverse=True)

    ranked = []
    for i, market in enumerate(sorted_markets[:top_n]):
        ranked.append(RankedMarket(
            rank=i + 1,
            market=market,
            rank_value=market.volume_24h
        ))

    return ranked


def rank_by_lowest_open_interest(
    markets: List[MarketInfo],
    top_n: int = 10,
    min_volume: float = 100
) -> List[RankedMarket]:
    """
    Rank markets by lowest open interest (for finding less crowded markets).

    Only includes markets with minimum volume to avoid dead markets.

    Args:
        markets: List of markets
        top_n: Number of top markets to return
        min_volume: Minimum 24h volume to include

    Returns:
        List of ranked markets (lowest OI first)
    """
    # Filter by minimum volume first
    active_markets = [m for m in markets if m.volume_24h >= min_volume]

    # For now, use total_volume as proxy for OI if OI not available
    # (In real implementation, we'd fetch actual OI from API)
    sorted_markets = sorted(active_markets, key=lambda m: m.total_volume)

    ranked = []
    for i, market in enumerate(sorted_markets[:top_n]):
        ranked.append(RankedMarket(
            rank=i + 1,
            market=market,
            rank_value=market.total_volume  # Using total volume as OI proxy
        ))

    return ranked


def rank_by_lowest_24h_volume(
    markets: List[MarketInfo],
    top_n: int = 10,
    min_volume: float = 100
) -> List[RankedMarket]:
    """
    Rank markets by lowest 24h volume (for finding quieter markets).

    Only includes markets with minimum volume to avoid dead/inactive markets.

    Args:
        markets: List of markets
        top_n: Number of top markets to return
        min_volume: Minimum 24h volume to include ($100 default)

    Returns:
        List of ranked markets (lowest 24h volume first)
    """
    # Filter by minimum volume first (to exclude dead markets)
    active_markets = [m for m in markets if m.volume_24h >= min_volume]

    # Sort by 24h volume ascending (lowest first)
    sorted_markets = sorted(active_markets, key=lambda m: m.volume_24h)

    ranked = []
    for i, market in enumerate(sorted_markets[:top_n]):
        ranked.append(RankedMarket(
            rank=i + 1,
            market=market,
            rank_value=market.volume_24h
        ))

    return ranked


def rank_by_highest_total_volume(
    markets: List[MarketInfo],
    top_n: int = 10
) -> List[RankedMarket]:
    """
    Rank markets by highest total volume (most traded overall).

    Args:
        markets: List of markets
        top_n: Number of top markets to return

    Returns:
        List of ranked markets (highest total volume first)
    """
    # Sort by total volume descending (highest first)
    sorted_markets = sorted(markets, key=lambda m: m.total_volume, reverse=True)

    ranked = []
    for i, market in enumerate(sorted_markets[:top_n]):
        ranked.append(RankedMarket(
            rank=i + 1,
            market=market,
            rank_value=market.total_volume
        ))

    return ranked


def rank_by_lowest_total_volume(
    markets: List[MarketInfo],
    top_n: int = 10,
    min_volume: float = 100
) -> List[RankedMarket]:
    """
    Rank markets by lowest total volume (less established markets).

    Only includes markets with minimum 24h volume to avoid dead markets.

    Args:
        markets: List of markets
        top_n: Number of top markets to return
        min_volume: Minimum 24h volume to include ($100 default)

    Returns:
        List of ranked markets (lowest total volume first)
    """
    # Filter by minimum 24h volume first (to exclude dead markets)
    active_markets = [m for m in markets if m.volume_24h >= min_volume]

    # Sort by total volume ascending (lowest first)
    sorted_markets = sorted(active_markets, key=lambda m: m.total_volume)

    ranked = []
    for i, market in enumerate(sorted_markets[:top_n]):
        ranked.append(RankedMarket(
            rank=i + 1,
            market=market,
            rank_value=market.total_volume
        ))

    return ranked


@dataclass
class EfficiencyRankedMarket:
    """A market with efficiency indicator ranking info."""
    rank: int
    market: MarketInfo
    efficiency_indicator: float  # volume_24h / orderbook_liquidity
    volume_24h: float
    orderbook_liquidity: float
    side: str  # "YES" or "NO" - which side has best efficiency


def rank_by_efficiency_indicator(
    client: OpinionClient,
    markets: List[MarketInfo],
    top_n: int = 10,
    bonus_points_only: bool = True
) -> List[EfficiencyRankedMarket]:
    """
    Rank markets by EfficiencyIndicator = volume_24h / orderbook_liquidity.

    Higher efficiency = more recent trading activity relative to current liquidity.
    This finds markets with high volume but low orderbook depth (easier to get fills).

    Args:
        client: Opinion client for fetching orderbooks
        markets: List of markets to analyze
        top_n: Number of top markets to return
        bonus_points_only: If True, only include markets with bonus points

    Returns:
        List of efficiency-ranked markets
    """
    from .utils.logging_utils import get_logger
    logger = get_logger()

    # Debug: Check how many markets have bonus points flag set
    bonus_count = sum(1 for m in markets if m.has_bonus_points)
    print(f"  Total markets: {len(markets)}, Markets with has_bonus_points=True: {bonus_count}")

    # Filter to bonus points markets if requested
    if bonus_points_only:
        eligible_markets = [m for m in markets if m.has_bonus_points]
        print(f"  Found {len(eligible_markets)} markets with Bonus Points")
    else:
        eligible_markets = markets

    if not eligible_markets:
        print("  No eligible markets found.")
        print("  Note: Bonus Points markets are those with 'Pre-TGE' label.")
        return []

    print(f"  Calculating EfficiencyIndicator for {len(eligible_markets)} markets...")
    print("  (This may take a moment as we fetch orderbook data)")

    efficiency_data = []

    for i, market in enumerate(eligible_markets):
        # Show progress
        if (i + 1) % 5 == 0 or i == 0:
            print(f"    Processing {i + 1}/{len(eligible_markets)}...")

        # Calculate efficiency for YES side
        yes_orderbook = client.get_orderbook(market.yes_token_id, market.question_id, 0)
        if yes_orderbook:
            yes_liquidity = float(yes_orderbook.total_liquidity())
            if yes_liquidity > 0:
                yes_efficiency = market.volume_24h / yes_liquidity
            else:
                yes_efficiency = float('inf') if market.volume_24h > 0 else 0
        else:
            yes_efficiency = 0
            yes_liquidity = 0

        # Calculate efficiency for NO side
        no_orderbook = client.get_orderbook(market.no_token_id, market.question_id, 1)
        if no_orderbook:
            no_liquidity = float(no_orderbook.total_liquidity())
            if no_liquidity > 0:
                no_efficiency = market.volume_24h / no_liquidity
            else:
                no_efficiency = float('inf') if market.volume_24h > 0 else 0
        else:
            no_efficiency = 0
            no_liquidity = 0

        # Use the side with better efficiency (higher ratio)
        if yes_efficiency >= no_efficiency:
            best_efficiency = yes_efficiency
            best_liquidity = yes_liquidity
            best_side = "YES"
        else:
            best_efficiency = no_efficiency
            best_liquidity = no_liquidity
            best_side = "NO"

        # Only include if we have valid data
        if best_efficiency > 0 and best_efficiency != float('inf'):
            efficiency_data.append({
                "market": market,
                "efficiency": best_efficiency,
                "volume_24h": market.volume_24h,
                "liquidity": best_liquidity,
                "side": best_side,
            })

    # Sort by efficiency (highest first)
    efficiency_data.sort(key=lambda x: x["efficiency"], reverse=True)

    # Build ranked list
    ranked = []
    for i, data in enumerate(efficiency_data[:top_n]):
        ranked.append(EfficiencyRankedMarket(
            rank=i + 1,
            market=data["market"],
            efficiency_indicator=data["efficiency"],
            volume_24h=data["volume_24h"],
            orderbook_liquidity=data["liquidity"],
            side=data["side"],
        ))

    return ranked


def display_efficiency_table(ranked_markets: List[EfficiencyRankedMarket]):
    """Display a formatted table of efficiency-ranked markets with full titles."""
    print("\n" + "=" * 100)
    print("  TOP MARKETS BY EFFICIENCY INDICATOR (Bonus Points Only)")
    print("  EfficiencyIndicator = 24h Volume / Orderbook Liquidity")
    print("  Bonus Points = Macro + Pre-TGE labeled markets")
    print("=" * 100)

    if not ranked_markets:
        print("  No markets found with bonus points.")
        print("=" * 100)
        return

    for rm in ranked_markets:
        m = rm.market
        # Show FULL title on its own line, no truncation
        print(f"\n  {rm.rank}. {m.title}")
        print(f"     Side: {rm.side} | 24h Vol: ${rm.volume_24h:,.0f} | "
              f"Liquidity: ${rm.orderbook_liquidity:,.0f} | Efficiency: {rm.efficiency_indicator:.2f}")

    print("\n" + "=" * 100)


def search_markets_by_keyword(
    markets: List[MarketInfo],
    keyword: str
) -> List[MarketInfo]:
    """
    Search markets by keyword (case-insensitive).

    Args:
        markets: List of markets to search
        keyword: Keyword to search for in market titles

    Returns:
        List of markets matching the keyword
    """
    keyword_lower = keyword.lower().strip()

    matching = []
    for market in markets:
        if keyword_lower in market.title.lower():
            matching.append(market)

    return matching


def keyword_search_interactive(markets: List[MarketInfo]) -> Optional[MarketInfo]:
    """
    Interactive keyword search for markets.

    Args:
        markets: List of all markets to search

    Returns:
        Selected MarketInfo or None if cancelled
    """
    print("\n" + "-" * 60)
    print("  KEYWORD SEARCH")
    print("-" * 60)
    print("  Enter keywords to search for markets (e.g., 'usdt depeg')")
    print("  Enter 'b' to go back to the main menu")
    print()

    while True:
        keyword = input("  Search: ").strip()

        if keyword.lower() == 'b':
            return None

        if not keyword:
            print("  Please enter a search term.")
            continue

        # Search for matching markets
        matching = search_markets_by_keyword(markets, keyword)

        if not matching:
            print(f"  No markets found matching '{keyword}'. Try different keywords.")
            continue

        if len(matching) == 1:
            # Only one match - confirm and return
            market = matching[0]
            print(f"\n  Found 1 market:")
            print(f"    {market.title}")
            print(f"    24h Volume: ${market.volume_24h:,.0f} | Total: ${market.total_volume:,.0f}")
            print()

            confirm = input("  Select this market? (y/n): ").strip().lower()
            if confirm == 'y':
                return market
            else:
                print("  Searching again...")
                continue
        else:
            # Multiple matches - show list and let user choose
            print(f"\n  Found {len(matching)} markets matching '{keyword}':")
            print("-" * 60)

            for i, market in enumerate(matching, 1):
                print(f"  {i}) {market.title}")
                print(f"     24h: ${market.volume_24h:,.0f} | Total: ${market.total_volume:,.0f}")

            print("-" * 60)
            print()

            while True:
                choice = input(f"  Enter number (1-{len(matching)}) or 'b' to search again: ").strip().lower()

                if choice == 'b':
                    break

                try:
                    idx = int(choice)
                    if 1 <= idx <= len(matching):
                        return matching[idx - 1]
                    else:
                        print(f"  Please enter a number between 1 and {len(matching)}")
                except ValueError:
                    print("  Invalid input. Enter a number or 'b'.")


def display_market_table(
    ranked_markets: List[RankedMarket],
    title: str,
    rank_label: str = "Value"
):
    """
    Display a formatted table of ranked markets with full titles.

    Args:
        ranked_markets: List of ranked markets
        title: Title for the table
        rank_label: Label for the ranking value column
    """
    logger = get_logger()

    # Header
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

    if not ranked_markets:
        print("  No markets found matching criteria.")
        print("=" * 100)
        return

    for rm in ranked_markets:
        m = rm.market
        # Show FULL title on its own line, no truncation
        print(f"\n  {rm.rank}. {m.title}")
        print(f"     24h Vol: ${m.volume_24h:,.0f} | {rank_label}: ${rm.rank_value:,.0f}")

    print("\n" + "=" * 100)


def market_id_search_interactive(client: OpinionClient) -> Optional[MarketInfo]:
    """
    Interactive market selection by ID or URL.

    Allows users to paste a market URL or enter a market ID directly.
    Handles multi-topic markets by showing all child options.

    Args:
        client: Opinion client instance

    Returns:
        Selected MarketInfo or None if cancelled
    """
    import re

    print("\n" + "-" * 60)
    print("  MARKET ID / URL SEARCH")
    print("-" * 60)
    print("  Enter a market ID (e.g., '55') or paste a URL like:")
    print("  https://app.opinion.trade/detail?topicId=55&type=multi")
    print("  Enter 'b' to go back to the main menu")
    print()

    while True:
        user_input = input("  Market ID or URL: ").strip()

        if user_input.lower() == 'b':
            return None

        if not user_input:
            print("  Please enter a market ID or URL.")
            continue

        # Try to extract market ID and type from input
        market_id = None
        is_multi = False

        # Check if it's a URL with topicId parameter
        url_match = re.search(r'topicId=(\d+)', user_input)
        if url_match:
            market_id = int(url_match.group(1))
            # Check if URL contains type=multi
            is_multi = 'type=multi' in user_input.lower()
        else:
            # Try to parse as a plain number
            try:
                market_id = int(user_input)
                # Assume multi-topic by default for better compatibility
                is_multi = True
            except ValueError:
                print("  Could not find a valid market ID. Enter a number or paste a URL.")
                continue

        print(f"  Fetching market ID {market_id}{'  (multi-topic)' if is_multi else ''}...")

        # Try to get the market - check if it's a multi-topic market
        # Use the /mutil/ endpoint for multi-topic markets (like Fed Rate Decision)
        children = client.get_market_children(market_id, is_multi=is_multi)

        if children:
            # Multi-topic market - show all children
            print(f"\n  Found multi-topic market with {len(children)} options:")
            print("-" * 60)

            for i, market in enumerate(children, 1):
                print(f"  {i}) {market.title}")
                print(f"     24h: ${market.volume_24h:,.0f} | Total: ${market.total_volume:,.0f}")

            print("-" * 60)
            print()

            while True:
                choice = input(f"  Enter number (1-{len(children)}) or 'b' to search again: ").strip().lower()

                if choice == 'b':
                    break

                try:
                    idx = int(choice)
                    if 1 <= idx <= len(children):
                        selected = children[idx - 1]
                        print(f"\n  Selected: {selected.title}")
                        return selected
                    else:
                        print(f"  Please enter a number between 1 and {len(children)}")
                except ValueError:
                    print("  Invalid input. Enter a number or 'b'.")
        else:
            # Single market or not found
            market = client.get_market(market_id)

            if market:
                print(f"\n  Found market:")
                print(f"    {market.title}")
                print(f"    24h Volume: ${market.volume_24h:,.0f} | Total: ${market.total_volume:,.0f}")
                print()

                confirm = input("  Select this market? (y/n): ").strip().lower()
                if confirm == 'y':
                    return market
                else:
                    print("  Searching again...")
            else:
                print(f"  Market ID {market_id} not found or not tradeable.")
                print("  Make sure the market exists and has tradeable options.")


def select_market_interactive(
    client: OpinionClient,
    filters: MarketFilters
) -> Optional[MarketInfo]:
    """
    Interactive market selection flow.

    Shows user ranked lists or keyword search to choose a market.

    Args:
        client: Opinion client instance
        filters: Market filter settings

    Returns:
        Selected MarketInfo or None if cancelled
    """
    logger = get_logger()

    print("\n" + "=" * 80)
    print("  MARKET DISCOVERY")
    print("=" * 80)
    print("  Fetching markets from Opinion.trade...")

    # Fetch all active markets
    all_markets = fetch_active_markets(client, limit=100)

    if not all_markets:
        print("  ERROR: Could not fetch markets. Check your connection.")
        return None

    print(f"  Found {len(all_markets)} active markets.")

    # Filter markets
    filtered = filter_markets(all_markets, filters)
    print(f"  After filtering: {len(filtered)} markets meet criteria.")

    if not filtered:
        print("  No markets pass the filters. Try lowering min_24h_volume in config.yaml")
        return None

    # Let user choose selection mode
    while True:
        print("\n  How would you like to find a market?")
        print("  1) By HIGHEST 24h volume (most active today)")
        print("  2) By LOWEST 24h volume (quietest today)")
        print("  3) By HIGHEST total volume (most traded overall)")
        print("  4) By LOWEST total volume (less established)")
        print("  5) By KEYWORD search (search by name)")
        print("  6) By MARKET ID or URL (direct access)")
        print("  7) By EFFICIENCY INDICATOR (Bonus Points markets only)")
        print("  q) Quit / Cancel")
        print()

        choice = input("  Enter 1-7 or q: ").strip().lower()

        if choice == 'q':
            return None

        if choice == '1':
            by_volume = rank_by_volume(filtered, top_n=filters.top_n_by_volume)
            display_market_table(by_volume, "TOP MARKETS BY HIGHEST 24H VOLUME", "24h Volume")
            selected_list = by_volume
            break
        elif choice == '2':
            by_low_24h = rank_by_lowest_24h_volume(
                filtered, top_n=filters.top_n_by_volume, min_volume=filters.min_24h_volume
            )
            display_market_table(by_low_24h, "MARKETS WITH LOWEST 24H VOLUME", "24h Volume")
            selected_list = by_low_24h
            break
        elif choice == '3':
            by_high_total = rank_by_highest_total_volume(filtered, top_n=filters.top_n_by_volume)
            display_market_table(by_high_total, "TOP MARKETS BY HIGHEST TOTAL VOLUME", "Total Vol")
            selected_list = by_high_total
            break
        elif choice == '4':
            by_low_total = rank_by_lowest_total_volume(
                filtered, top_n=filters.top_n_by_volume, min_volume=filters.min_24h_volume
            )
            display_market_table(by_low_total, "MARKETS WITH LOWEST TOTAL VOLUME", "Total Vol")
            selected_list = by_low_total
            break
        elif choice == '5':
            # Keyword search mode
            result = keyword_search_interactive(all_markets)
            if result:
                return result
            # If user cancelled keyword search, show menu again
            continue
        elif choice == '6':
            # Direct market ID or URL mode
            result = market_id_search_interactive(client)
            if result:
                return result
            # If user cancelled, show menu again
            continue
        elif choice == '7':
            # Efficiency Indicator mode (Bonus Points markets only)
            efficiency_ranked = rank_by_efficiency_indicator(
                client, all_markets, top_n=filters.top_n_by_volume, bonus_points_only=True
            )
            display_efficiency_table(efficiency_ranked)

            if not efficiency_ranked:
                print("  No bonus points markets found. Try another filter.")
                continue

            # Let user select from efficiency list
            while True:
                eff_choice = input(f"  Enter number (1-{len(efficiency_ranked)}) or 'b' to go back: ").strip().lower()

                if eff_choice == 'b':
                    break

                try:
                    idx = int(eff_choice)
                    if 1 <= idx <= len(efficiency_ranked):
                        selected = efficiency_ranked[idx - 1]
                        print(f"\n  Selected: {selected.market.title}")
                        print(f"  Recommended side: {selected.side}")
                        return selected.market
                    else:
                        print(f"  Please enter a number between 1 and {len(efficiency_ranked)}")
                except ValueError:
                    print("  Invalid input. Enter a number or 'b'.")
            continue
        else:
            print("  Invalid choice. Please enter 1-7 or q.")

    # Let user select a specific market
    while True:
        print()
        choice = input(f"  Enter market number (1-{len(selected_list)}) or 'b' to go back: ").strip().lower()

        if choice == 'b':
            # Go back to list selection
            return select_market_interactive(client, filters)

        try:
            market_num = int(choice)
            if 1 <= market_num <= len(selected_list):
                selected = selected_list[market_num - 1].market
                print(f"\n  Selected: {selected.title}")
                return selected
            else:
                print(f"  Please enter a number between 1 and {len(selected_list)}")
        except ValueError:
            print("  Invalid input. Enter a number or 'b' to go back.")


def select_side_interactive() -> Optional[str]:
    """
    Let user choose YES or NO side.

    Returns:
        "YES", "NO", or None if cancelled
    """
    print("\n" + "-" * 40)
    print("  Which side do you want to trade?")
    print("-" * 40)
    print("  1) YES - Buy YES shares (bet that outcome happens)")
    print("  2) NO  - Buy NO shares (bet that outcome doesn't happen)")
    print("  q) Cancel")
    print()

    while True:
        choice = input("  Enter 1, 2, or q: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == '1':
            print("  Selected: YES side")
            return "YES"
        elif choice == '2':
            print("  Selected: NO side")
            return "NO"
        else:
            print("  Invalid choice. Please enter 1, 2, or q.")
