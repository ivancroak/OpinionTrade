"""
Opinion.trade client with SIWE authentication.

This module provides a client that:
- Authenticates using Sign-In With Ethereum (SIWE)
- Gets a bearer token from the API
- Uses the bearer token for all subsequent requests
- No API key required
"""

import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from string import hexdigits

import aiohttp
import requests
from eth_account import Account
from eth_account.messages import encode_defunct, encode_typed_data


# ===========================================
# EIP-712 Typed Data for Order Signing
# ===========================================

# This is the typed data structure required by Opinion.trade for order signing
# All orders must be signed with this EIP-712 structure
ORDER_TYPED_DATA = {
    "primaryType": "Order",
    "types": {
        "EIP712Domain": [
            {"name": "name", "type": "string"},
            {"name": "version", "type": "string"},
            {"name": "chainId", "type": "uint256"},
            {"name": "verifyingContract", "type": "address"}
        ],
        "Order": [
            {"name": "salt", "type": "uint256"},
            {"name": "maker", "type": "address"},
            {"name": "signer", "type": "address"},
            {"name": "taker", "type": "address"},
            {"name": "tokenId", "type": "uint256"},
            {"name": "makerAmount", "type": "uint256"},
            {"name": "takerAmount", "type": "uint256"},
            {"name": "expiration", "type": "uint256"},
            {"name": "nonce", "type": "uint256"},
            {"name": "feeRateBps", "type": "uint256"},
            {"name": "side", "type": "uint8"},
            {"name": "signatureType", "type": "uint8"}
        ]
    },
    "domain": {
        "name": "OPINION CTF Exchange",
        "version": "1",
        "chainId": 56,
        "verifyingContract": "0x5f45344126d6488025b0b84a3a8189f2487a7246"
    },
    "message": {
        # Default values - will be overridden per order
        "taker": "0x0000000000000000000000000000000000000000",
        "expiration": "0",
        "nonce": "0",
        "feeRateBps": "0",
        "signatureType": "2",
    },
}

# USDT contract address on BSC
USDT_CONTRACT_ADDRESS = "0x55d398326f99059fF775485246999027B3197955"

from .utils.logging_utils import get_logger, log_wallet
from .config_loader import WalletConfig, NetworkConfig


# ===========================================
# Data Classes for Clean API
# ===========================================

@dataclass
class OrderbookLevel:
    """One price level in the orderbook."""
    price: Decimal
    amount: Decimal


@dataclass
class Orderbook:
    """Full orderbook with bids and asks."""
    bids: List[OrderbookLevel]  # Sorted: highest price first
    asks: List[OrderbookLevel]  # Sorted: lowest price first

    def best_bid(self) -> Optional[Decimal]:
        """Get the best (highest) bid price."""
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[Decimal]:
        """Get the best (lowest) ask price."""
        return self.asks[0].price if self.asks else None

    def spread(self) -> Optional[Decimal]:
        """Get the bid-ask spread."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb and ba:
            return ba - bb
        return None

    def total_bid_liquidity(self) -> Decimal:
        """Get total USD value of all bid orders."""
        total = Decimal("0")
        for level in self.bids:
            # amount is in shares, price is per share, so amount * price = USD value
            total += level.amount * level.price
        return total

    def total_ask_liquidity(self) -> Decimal:
        """Get total USD value of all ask orders."""
        total = Decimal("0")
        for level in self.asks:
            total += level.amount * level.price
        return total

    def total_liquidity(self) -> Decimal:
        """Get total USD liquidity (bids + asks)."""
        return self.total_bid_liquidity() + self.total_ask_liquidity()


@dataclass
class MarketInfo:
    """Information about a prediction market."""
    market_id: int
    title: str
    status: str
    yes_token_id: str
    no_token_id: str
    volume_24h: float
    total_volume: float
    condition_id: str
    chain_id: int
    question_id: str = ""

    # Bonus points indicator - markets with active incentives
    has_bonus_points: bool = False

    # Parent topic ID for multi-topic markets (needed for orderbook)
    parent_topic_id: int = 0

    # These might not be available from API, set defaults
    tick_size: Decimal = Decimal("0.001")  # Default 0.1 cent tick (Opinion.trade uses 0.001 increments)
    min_order_size: Decimal = Decimal("1")  # Default $1 minimum


@dataclass
class OrderInfo:
    """Information about an order."""
    order_id: str
    market_id: int
    token_id: str
    side: str  # "BUY" or "SELL"
    price: Decimal
    original_size: Decimal
    filled_size: Decimal
    remaining_size: Decimal
    status: str  # "open", "filled", "cancelled", etc.

    def is_fully_filled(self) -> bool:
        return self.remaining_size <= 0

    def is_partially_filled(self) -> bool:
        return self.filled_size > 0 and self.remaining_size > 0


@dataclass
class PositionInfo:
    """Information about a position."""
    market_id: int
    token_id: str
    side: str  # "YES" or "NO"
    shares: Decimal
    avg_price: Decimal
    current_value: Decimal
    market_title: str = ""  # Full market title (e.g., "Bitcoin above 86,000 on January 5?")


@dataclass
class BalanceInfo:
    """Wallet balance information."""
    available: Decimal
    frozen: Decimal  # Locked in open orders
    total: Decimal


# ===========================================
# Opinion Client Class with SIWE Auth
# ===========================================

class OpinionClient:
    """
    Opinion.trade client using SIWE (Sign-In With Ethereum) authentication.

    Authentication flow:
    1. Build SIWE message with wallet address, nonce, timestamp
    2. Sign message with private key
    3. POST to /api/bsc/api/v1/user/token
    4. Store returned bearer token
    5. Use bearer token for all API calls
    """

    BASE_URL = "https://proxy.opinion.trade:8443"

    def __init__(
        self,
        wallet: WalletConfig,
        network: NetworkConfig,
    ):
        """
        Initialize the Opinion client for a specific wallet.

        Args:
            wallet: Wallet configuration (private key, proxy, etc.)
            network: Network configuration (RPC URL, chain ID, etc.)
        """
        self.wallet = wallet
        self.network = network
        self.logger = get_logger()

        # Authentication state
        self._bearer_token: Optional[str] = None
        self._connected = False
        self._address: Optional[str] = None
        self._proxy_wallet: Optional[str] = None  # Multi-sig wallet from profile

        # Derive address from private key immediately
        try:
            account = Account.from_key(self.wallet.private_key)
            self._address = account.address
            self.wallet.address = self._address
            self._account = account
        except Exception as e:
            self.logger.error(f"Invalid private key: {e}")
            self._account = None

        # Session for HTTP requests
        self._session: Optional[requests.Session] = None

    @property
    def wallet_id(self) -> int:
        return self.wallet.wallet_id

    @property
    def address(self) -> Optional[str]:
        return self._address

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Origin": "https://app.opinion.trade",
            "Referer": "https://app.opinion.trade/",
            "Content-Type": "application/json",
            "x-device-kind": "web",
            "x-device-fingerprint": "".join(random.choices(hexdigits, k=32)).lower(),
        }
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        return headers

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """Get proxy configuration for requests."""
        return self.wallet.get_proxy_dict()

    def _build_siwe_message(self, nonce: int, timestamp: datetime) -> str:
        """
        Build the SIWE (Sign-In With Ethereum) message.

        This message format is required by Opinion.trade for authentication.
        Format matches exactly what the reference bot uses.
        """
        # Reference bot uses: date_now.isoformat()[:-9] + 'Z'
        # This removes the last 9 chars (part of microseconds + timezone) and adds Z
        iso_time = timestamp.isoformat()[:-9] + 'Z'

        message = f"""app.opinion.trade wants you to sign in with your Ethereum account:
{self._address}

Welcome to opinion.trade! By proceeding, you agree to our Privacy Policy and Terms of Use.

URI: https://app.opinion.trade
Version: 1
Chain ID: 56
Nonce: {nonce}
Issued At: {iso_time}"""

        return message

    def _sign_message(self, message: str) -> str:
        """
        Sign a message with the wallet's private key.

        Returns signature as hex string WITHOUT 0x prefix (as required by Opinion API).
        """
        encoded = encode_defunct(text=message)
        signed = self._account.sign_message(encoded)
        signature = signed.signature.hex()
        # Remove 0x prefix - Opinion API expects signature without it
        if signature.startswith('0x'):
            signature = signature[2:]
        return signature

    def _sign_typed_data(self, typed_data: dict) -> str:
        """
        Sign EIP-712 typed data for order placement.

        Args:
            typed_data: Complete EIP-712 typed data structure

        Returns:
            Signature as hex string WITH 0x prefix (as required for orders)
        """
        encoded = encode_typed_data(full_message=typed_data)
        signed = self._account.sign_message(encoded)
        signature = signed.signature.hex()
        # Orders require 0x prefix
        if not signature.startswith('0x'):
            signature = '0x' + signature
        return signature

    def _build_order_typed_data(
        self,
        token_id: str,
        maker_amount: Decimal,
        taker_amount: Decimal,
        side: int  # 0 = BUY, 1 = SELL
    ) -> dict:
        """
        Build the EIP-712 typed data structure for an order.

        Args:
            token_id: The token ID (YES or NO token)
            maker_amount: Amount in USDT (for BUY) or tokens (for SELL)
            taker_amount: Amount of tokens (for BUY) or USDT (for SELL)
            side: 0 for BUY, 1 for SELL

        Returns:
            Complete typed data dict ready for signing
        """
        import copy

        # Deep copy the template
        typed_data = copy.deepcopy(ORDER_TYPED_DATA)

        # Generate salt (random number based on time)
        salt = str(int(random.random() * int(time.time() * 1000)))

        # Convert amounts to wei (× 1e18)
        maker_amount_wei = str(int(maker_amount * Decimal('1e18')))
        taker_amount_wei = str(int(taker_amount * Decimal('1e18')))

        # Update message with order-specific values
        typed_data["message"].update({
            "salt": salt,
            "maker": self._proxy_wallet,  # Multi-sig wallet
            "signer": self._address,       # Actual wallet
            "tokenId": token_id,
            "makerAmount": maker_amount_wei,
            "takerAmount": taker_amount_wei,
            "side": str(side),
        })

        return typed_data

    def connect(self) -> bool:
        """
        Connect to Opinion.trade using SIWE authentication.

        1. Check if user is registered
        2. Build and sign SIWE message
        3. Get bearer token from API
        4. Store token for subsequent requests

        Returns:
            True if connection successful, False otherwise
        """
        if not self._account:
            self.logger.error("No valid account - check private key")
            return False

        log_wallet(self.wallet_id, f"Connecting to Opinion.trade...")
        log_wallet(self.wallet_id, f"Wallet address: {self._address}")

        try:
            # Create session
            self._session = requests.Session()
            self._session.headers.update(self._get_headers())

            proxies = self._get_proxies()

            # Step 1: Check if user is registered
            check_url = f"{self.BASE_URL}/api/bsc/api/v1/user/is/new/user?wallet_address={self._address}"
            resp = self._session.get(check_url, proxies=proxies, timeout=30)
            check_data = resp.json()

            if not check_data.get("result") or "result" not in check_data.get("result", {}):
                self.logger.error(f"Failed to check registration: {check_data}")
                return False

            is_new_user = check_data["result"]["result"]
            if is_new_user:
                self.logger.error(f"User not registered on Opinion.trade. Please register via web first.")
                return False

            # Step 2: Build SIWE message
            date_now = datetime.now(timezone.utc)
            nonce = random.randint(65535, 0xffffffffffff)
            siwe_message = self._build_siwe_message(nonce, date_now)

            # Step 3: Sign the message
            signature = self._sign_message(siwe_message)

            # Step 4: Get bearer token
            login_url = f"{self.BASE_URL}/api/bsc/api/v1/user/token"
            login_payload = {
                "nonce": str(nonce),
                "timestamp": int(date_now.timestamp()),
                "siwe_message": siwe_message,
                "sign": signature,
                "invite_code": "",
                "sources": "web",
                "sign_in_wallet_plugin": None
            }

            resp = self._session.post(login_url, json=login_payload, proxies=proxies, timeout=30)
            login_data = resp.json()

            if login_data.get("errmsg") or login_data.get("errno"):
                self.logger.error(f"Login failed: {login_data}")
                return False

            # Step 5: Store bearer token
            self._bearer_token = login_data["result"]["token"]
            self._session.headers["Authorization"] = f"Bearer {self._bearer_token}"

            # Step 6: Get profile info to verify and get proxy wallet
            profile_url = f"{self.BASE_URL}/api/bsc/api/v2/user/{self._address}/profile?chainId=56"
            resp = self._session.get(profile_url, proxies=proxies, timeout=30)
            profile_data = resp.json()

            if profile_data.get("errmsg") or profile_data.get("errno"):
                self.logger.error(f"Failed to get profile: {profile_data}")
                return False

            profile = profile_data["result"]
            self._proxy_wallet = profile.get("multiSignedWalletAddress", {}).get("56")

            if not self._proxy_wallet:
                self.logger.warning("No proxy wallet found - some features may not work")

            self._connected = True
            log_wallet(self.wallet_id, f"Connected! Bearer token obtained.")

            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error during connection: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect and clean up."""
        if self._session:
            self._session.close()
        self._session = None
        self._bearer_token = None
        self._connected = False
        log_wallet(self.wallet_id, "Disconnected")

    def _request(self, method: str, endpoint: str, retries: int = 3, **kwargs) -> Optional[dict]:
        """
        Make an authenticated request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (will be appended to BASE_URL)
            retries: Number of retries on failure (default 3)
            **kwargs: Additional arguments for requests

        Returns:
            JSON response or None on error
        """
        if not self._connected or not self._session:
            self.logger.error("Not connected - call connect() first")
            return None

        url = f"{self.BASE_URL}{endpoint}"
        proxies = self._get_proxies()

        last_error = None
        for attempt in range(retries):
            try:
                resp = self._session.request(method, url, proxies=proxies, timeout=30, **kwargs)

                # Check for empty response
                if not resp.content or len(resp.content) == 0:
                    self.logger.warning(f"Empty response from API (status {resp.status_code})")
                    last_error = "Empty response"
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        continue
                    return None

                # Check for non-200 status
                if resp.status_code != 200:
                    self.logger.warning(f"API returned status {resp.status_code}")
                    last_error = f"Status {resp.status_code}"
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                # Try to parse JSON
                try:
                    data = resp.json()
                except ValueError as json_err:
                    self.logger.warning(f"Invalid JSON response: {str(resp.content[:100])}")
                    last_error = f"Invalid JSON: {json_err}"
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return None

                if data.get("errno") and data.get("errno") != 0:
                    self.logger.error(f"API error: {data.get('errmsg', 'Unknown error')}")
                    return None

                return data

            except requests.exceptions.ProxyError as e:
                self.logger.error(f"Proxy error: {e}")
                last_error = f"Proxy error: {e}"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"Connection error: {e}")
                last_error = f"Connection error: {e}"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            except requests.exceptions.Timeout as e:
                self.logger.error(f"Request timeout: {e}")
                last_error = f"Timeout: {e}"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

            except Exception as e:
                self.logger.error(f"Request failed: {e}")
                last_error = str(e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        self.logger.error(f"All {retries} retries failed. Last error: {last_error}")
        return None

    # ===========================================
    # Market Data Methods
    # ===========================================

    def get_markets(
        self,
        status: str = "activated",
        limit: int = 100,
        keywords: str = ""
    ) -> List[MarketInfo]:
        """
        Fetch list of markets from Opinion.trade.

        Handles both single markets and multi-topic markets (with childList).
        Only returns markets that have valid YES/NO token IDs (tradeable markets).

        Args:
            status: "activated" for active markets, "resolved" for resolved
            limit: Maximum number of markets to fetch (default 100)
            keywords: Optional keyword filter for market titles
        """
        all_markets = []

        # Fetch both Binary (topicType=2) and Categorical (topicType=1) markets
        # topicType=2 appears to be binary markets, topicType=1 is categorical
        # Also fetch both status=1 (multi-topic parents) and status=2 (single/child markets)
        # Status 1 markets include Macro indicator markets like Fed Rate Decisions
        statuses = [1, 2] if status == "activated" else [4]

        for market_status in statuses:
            for topic_type in [1, 2]:
                params = {
                    "labelId": "",
                    "keywords": keywords,
                    "sortBy": 3,  # Sort by volume
                    "chainId": 56,
                    "limit": limit,
                    "status": market_status,
                    "isShow": "",  # Empty to get all markets including hidden
                    "topicType": topic_type,
                    "page": 1,
                    "indicatorType": "",  # Empty to get all indicator types
                }

                data = self._request("GET", "/api/bsc/api/v2/topic", params=params)
                if not data or "result" not in data:
                    continue

                all_markets.extend(self._parse_markets_response(data))

        # Remove duplicates by market_id
        seen_ids = set()
        unique_markets = []
        for m in all_markets:
            if m.market_id not in seen_ids:
                seen_ids.add(m.market_id)
                unique_markets.append(m)

        self.logger.debug(f"Fetched {len(unique_markets)} tradeable markets")
        return unique_markets

    def _parse_markets_response(self, data: dict) -> List[MarketInfo]:
        """Parse markets from API response, handling both single and multi-topic markets."""
        markets = []

        for m in data["result"].get("list", []):
            # Check if this is a multi-topic (has childList)
            child_list = m.get("childList", [])

            # Check for bonus points - can be in different fields
            # Common field names: incentiveFactor, bonusPoints, hasBonus, incentive
            has_bonus = self._check_bonus_points(m)
            parent_topic_id = m.get("topicId", 0)

            if child_list:
                # Multi-topic: expand each child as a separate market
                parent_title = m.get("title", "")
                for child in child_list:
                    # Skip if no token IDs (not tradeable)
                    if not child.get("yesPos") or not child.get("noPos"):
                        continue

                    # Combine parent + child title
                    child_title = child.get("title", "Unknown")
                    full_title = f"{parent_title} - {child_title}" if parent_title else child_title

                    # Child inherits bonus status from parent
                    child_has_bonus = has_bonus or self._check_bonus_points(child)

                    markets.append(MarketInfo(
                        market_id=child.get("topicId", 0),
                        title=full_title,
                        status=child.get("statusText", "unknown"),
                        yes_token_id=child.get("yesPos", ""),
                        no_token_id=child.get("noPos", ""),
                        volume_24h=float(child.get("volume24h", 0) or 0),
                        total_volume=float(child.get("volume", 0) or 0),
                        condition_id=child.get("conditionId", ""),
                        chain_id=child.get("chainId", 56),
                        question_id=child.get("questionId", ""),
                        has_bonus_points=child_has_bonus,
                        parent_topic_id=parent_topic_id,
                    ))
            else:
                # Single market: skip if no token IDs (not tradeable)
                if not m.get("yesPos") or not m.get("noPos"):
                    continue

                markets.append(MarketInfo(
                    market_id=m.get("topicId", 0),
                    title=m.get("title", "Unknown"),
                    status=m.get("statusText", "unknown"),
                    yes_token_id=m.get("yesPos", ""),
                    no_token_id=m.get("noPos", ""),
                    volume_24h=float(m.get("volume24h", 0) or 0),
                    total_volume=float(m.get("volume", 0) or 0),
                    condition_id=m.get("conditionId", ""),
                    chain_id=m.get("chainId", 56),
                    question_id=m.get("questionId", ""),
                    has_bonus_points=has_bonus,
                    parent_topic_id=0,
                ))

        return markets

    def _check_bonus_points(self, market_data: dict) -> bool:
        """
        Check if a market has bonus points active.

        Opinion.trade uses labels with high weights to indicate markets with
        bonus point incentives:
        - Macro (labelId=100002, weight=11) - Economic indicator markets
        - Pre-TGE (labelId=100007, weight=10) - Pre-token generation event markets

        These designated markets receive higher reward weights in the points system.
        """
        # Bonus point label IDs (labels with weight >= 10)
        BONUS_LABEL_IDS = {100002, 100007}  # Macro, Pre-TGE
        BONUS_LABEL_NAMES = {"Macro", "Pre-TGE"}

        # Check labelId array
        label_ids = market_data.get("labelId", [])
        if isinstance(label_ids, list):
            for lid in label_ids:
                if lid in BONUS_LABEL_IDS:
                    return True

        # Also check labelName as fallback
        label_names = market_data.get("labelName", [])
        if isinstance(label_names, list):
            for name in label_names:
                if name in BONUS_LABEL_NAMES:
                    return True

        return False

    def get_market(self, market_id: int) -> Optional[MarketInfo]:
        """
        Fetch details for a specific market.

        For multi-topic markets, returns the first tradeable child market.
        Use get_market_children() to get all child markets.
        """
        data = self._request("GET", f"/api/bsc/api/v2/topic/{market_id}")
        if not data or "result" not in data:
            return None

        m = data["result"].get("data", {})

        # Check if this is a multi-topic market with children
        child_list = m.get("childList", [])
        if child_list:
            # Return the first tradeable child
            parent_title = m.get("title", "")
            for child in child_list:
                if child.get("yesPos") and child.get("noPos"):
                    child_title = child.get("title", "Unknown")
                    full_title = f"{parent_title} - {child_title}" if parent_title else child_title
                    return MarketInfo(
                        market_id=child.get("topicId", 0),
                        title=full_title,
                        status=child.get("statusText", "unknown"),
                        yes_token_id=child.get("yesPos", ""),
                        no_token_id=child.get("noPos", ""),
                        volume_24h=float(child.get("volume24h", 0) or 0),
                        total_volume=float(child.get("volume", 0) or 0),
                        condition_id=child.get("conditionId", ""),
                        chain_id=child.get("chainId", 56),
                        question_id=child.get("questionId", ""),
                    )
            return None  # No tradeable children

        # Single market
        if not m.get("yesPos") or not m.get("noPos"):
            return None  # Not tradeable

        return MarketInfo(
            market_id=m.get("topicId", market_id),
            title=m.get("title", "Unknown"),
            status=m.get("statusText", "unknown"),
            yes_token_id=m.get("yesPos", ""),
            no_token_id=m.get("noPos", ""),
            volume_24h=float(m.get("volume24h", 0) or 0),
            total_volume=float(m.get("volume", 0) or 0),
            condition_id=m.get("conditionId", ""),
            chain_id=m.get("chainId", 56),
            question_id=m.get("questionId", ""),
        )

    def get_market_children(self, market_id: int, is_multi: bool = True) -> List[MarketInfo]:
        """
        Fetch all child markets for a multi-topic market.

        Args:
            market_id: Parent market ID (e.g., 55 for "US Fed Rate Decision in Dec?")
            is_multi: If True, use the /mutil/ endpoint for multi-topic markets

        Returns:
            List of child MarketInfo objects, or empty list if not multi-topic
        """
        # For multi-topic markets, Opinion.trade uses a different endpoint: /topic/mutil/
        # Note: "mutil" is the actual API endpoint name (not a typo)
        if is_multi:
            data = self._request("GET", f"/api/bsc/api/v2/topic/mutil/{market_id}")
        else:
            data = self._request("GET", f"/api/bsc/api/v2/topic/{market_id}")

        if not data or "result" not in data:
            # If mutil endpoint fails, try the regular endpoint
            if is_multi:
                return self.get_market_children(market_id, is_multi=False)
            return []

        m = data["result"].get("data", {})
        parent_title = m.get("title", "")
        child_list = m.get("childList", [])

        markets = []
        for child in child_list:
            if not child.get("yesPos") or not child.get("noPos"):
                continue

            child_title = child.get("title", "Unknown")
            full_title = f"{parent_title} - {child_title}" if parent_title else child_title

            markets.append(MarketInfo(
                market_id=child.get("topicId", 0),
                title=full_title,
                status=child.get("statusText", "unknown"),
                yes_token_id=child.get("yesPos", ""),
                no_token_id=child.get("noPos", ""),
                volume_24h=float(child.get("volume24h", 0) or 0),
                total_volume=float(child.get("volume", 0) or 0),
                condition_id=child.get("conditionId", ""),
                chain_id=child.get("chainId", 56),
                question_id=child.get("questionId", ""),
            ))

        return markets

    def get_orderbook(self, token_id: str, question_id: str = "", symbol_type: int = 0) -> Optional[Orderbook]:
        """
        Fetch the orderbook for a specific token (YES or NO side).
        """
        params = {
            "symbol_types": str(symbol_type),
            "question_id": question_id,
            "symbol": token_id,
            "chainId": "56",
        }

        data = self._request("GET", "/api/bsc/api/v2/order/market/depth", params=params)
        if not data or "result" not in data:
            return None

        book = data["result"]

        # Parse bids (buy orders) - sorted highest first
        bids = []
        for b in sorted(book.get("bids", []), key=lambda x: float(x[0]), reverse=True):
            bids.append(OrderbookLevel(
                price=Decimal(str(b[0])),
                amount=Decimal(str(b[1]))
            ))

        # Parse asks (sell orders) - sorted lowest first
        asks = []
        for a in sorted(book.get("asks", []), key=lambda x: float(x[0])):
            asks.append(OrderbookLevel(
                price=Decimal(str(a[0])),
                amount=Decimal(str(a[1]))
            ))

        return Orderbook(bids=bids, asks=asks)

    # ===========================================
    # Balance & Position Methods
    # ===========================================

    def get_balance(self) -> Optional[BalanceInfo]:
        """Get wallet balance (USDT)."""
        data = self._request("GET", f"/api/bsc/api/v2/user/{self._address}/profile", params={"chainId": 56})
        if not data or "result" not in data:
            return None

        profile = data["result"]
        balances = profile.get("balance", [])

        if not balances:
            return BalanceInfo(available=Decimal("0"), frozen=Decimal("0"), total=Decimal("0"))

        b = balances[0]
        available = Decimal(str(b.get("balance", 0)))
        frozen = Decimal(str(b.get("frozen", 0)))

        return BalanceInfo(
            available=available,
            frozen=frozen,
            total=available + frozen
        )

    def get_positions(self, market_id: int = 0) -> List[PositionInfo]:
        """Get open positions."""
        params = {
            "page": 1,
            "limit": 100,
            "walletAddress": self._address,
            "chainId": "56",
        }
        if market_id:
            params["topicId"] = market_id

        data = self._request("GET", "/api/bsc/api/v2/portfolio", params=params)
        if not data or "result" not in data:
            return []

        positions = []
        for p in data["result"].get("list", []):
            # API returns positionAvgPrice, not avgPrice
            avg_price = Decimal(str(p.get("positionAvgPrice", 0) or 0))
            shares = Decimal(str(p.get("tokenAmount", 0) or 0))

            # Build market title from API fields
            # mutilTitle = parent market (e.g., "Bitcoin above ... on January 5?")
            # topicTitle = specific option (e.g., "86,000")
            mutil_title = p.get("mutilTitle", "") or ""
            topic_title = p.get("topicTitle", "") or ""

            if mutil_title and topic_title:
                # Combine parent + child title
                # Replace "..." placeholder with the specific value if present
                if "..." in mutil_title:
                    market_title = mutil_title.replace("...", topic_title)
                else:
                    market_title = f"{mutil_title} - {topic_title}"
            elif mutil_title:
                market_title = mutil_title
            elif topic_title:
                market_title = topic_title
            else:
                market_title = f"Market #{p.get('topicId', 'Unknown')}"

            positions.append(PositionInfo(
                market_id=p.get("topicId", 0),
                token_id=p.get("tokenId", ""),
                side="YES" if p.get("outcomeSide") == 1 else "NO",
                shares=shares,
                avg_price=avg_price,
                current_value=Decimal(str(p.get("value", 0) or 0)),
                market_title=market_title
            ))

        return positions

    def get_my_orders(
        self,
        market_id: int = 0,
        order_type: str = "limit",
        limit: int = 50
    ) -> List[OrderInfo]:
        """Get list of my orders."""
        params = {
            "page": 1,
            "limit": limit,
            "walletAddress": self._address,
            "queryType": 1 if order_type == "limit" else 2,
        }
        if market_id:
            params["topicId"] = market_id

        data = self._request("GET", "/api/bsc/api/v2/order", params=params)
        if not data or "result" not in data:
            return []

        orders = []
        for o in data["result"].get("list", []):
            filled_parts = o.get("filled", "0/0").split("/")
            filled = Decimal(str(filled_parts[0])) if filled_parts else Decimal("0")
            total = Decimal(str(filled_parts[1])) if len(filled_parts) > 1 else Decimal("0")

            orders.append(OrderInfo(
                order_id=o.get("transNo", ""),
                market_id=o.get("topicId", 0),
                token_id=o.get("tokenId", ""),
                side="BUY" if o.get("side") == 1 else "SELL",
                price=Decimal(str(o.get("price", 0))),
                original_size=total,
                filled_size=filled,
                remaining_size=total - filled,
                status=o.get("status", "unknown")
            ))

        return orders

    def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """
        Get a specific order by its ID.

        Args:
            order_id: The order's transaction number

        Returns:
            OrderInfo if found, None otherwise
        """
        try:
            # Fetch all limit orders and find the one with matching ID
            orders = self.get_my_orders(order_type="limit", limit=100)

            if orders:  # Safeguard against None
                for order in orders:
                    if order.order_id == order_id:
                        return order

            # Also check market orders
            orders = self.get_my_orders(order_type="market", limit=100)

            if orders:  # Safeguard against None
                for order in orders:
                    if order.order_id == order_id:
                        return order

        except Exception as e:
            self.logger.error(f"Error fetching order {order_id}: {e}")

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        payload = {
            "trans_no": order_id,
            "chainId": 56,
        }

        data = self._request("POST", "/api/bsc/api/v1/order/cancel/order", json=payload)
        if not data:
            return False

        result = data.get("result", {})
        success = result.get("result", False)

        if success:
            log_wallet(self.wallet_id, f"Order cancelled: {order_id[:16]}...")

        return success

    def enable_trading(self) -> bool:
        """
        Check if trading is enabled (wallet is approved).

        Returns True if the proxy wallet is approved for trading.
        """
        if not self._proxy_wallet:
            self.logger.warning("No proxy wallet - cannot verify trading status")
            return True  # Assume enabled

        data = self._request(
            "GET",
            f"/api/bsc/api/v2/gnosis_safe/{self._proxy_wallet}/approved",
            params={"chainId": 56}
        )

        if not data:
            return False

        return data.get("result", False)

    # ===========================================
    # Order Placement
    # ===========================================

    def _submit_order(
        self,
        typed_data: dict,
        signature: str,
        market_id: int,
        price: Decimal,
        is_market_order: bool = False
    ) -> Optional[str]:
        """
        Submit a signed order to the API.

        Args:
            typed_data: The signed typed data structure
            signature: EIP-712 signature
            market_id: Topic/market ID
            price: Order price (0 for market orders)
            is_market_order: Whether this is a market order

        Returns:
            Transaction number (order ID) if successful, None otherwise
        """
        message = typed_data["message"]

        # Format price for API - must match exact format reference bot uses
        # Reference bot: price=str(price) where price is a float like 0.968
        if is_market_order:
            price_str = "0"
        else:
            # Truncate to 3 decimal places (not round) and convert to string
            price_float = float(price)
            price_truncated = int(price_float * 1000) / 1000
            price_str = str(price_truncated)

        payload = {
            "contractAddress": "",
            "orderExpTime": "0",
            "currencyAddress": USDT_CONTRACT_ADDRESS,
            "chainId": 56,
            # Include all message fields
            "salt": message["salt"],
            "maker": message["maker"],
            "signer": message["signer"],
            "taker": message["taker"],
            "tokenId": message["tokenId"],
            "makerAmount": message["makerAmount"],
            "takerAmount": message["takerAmount"],
            "expiration": message["expiration"],
            "nonce": message["nonce"],
            "feeRateBps": message["feeRateBps"],
            "side": message["side"],
            "signatureType": message["signatureType"],
            # Order-specific fields
            "topicId": market_id,
            "signature": signature,
            "sign": signature,
            "timestamp": int(time.time()),
            "safeRate": "0" if (message["side"] == "0" and is_market_order) else "0.05",
            "price": price_str,
            "tradingMethod": 1 if is_market_order else 2,  # 1 = market, 2 = limit
        }

        self.logger.debug(f"Order payload: price={price_str}, tokenId={message['tokenId']}, "
                          f"maker={message['maker']}, side={message['side']}")

        data = self._request("POST", "/api/bsc/api/v2/order", json=payload)

        if not data:
            self.logger.error("No response from order API")
            return None

        if "result" not in data:
            self.logger.error(f"Order API response: {data}")
            return None

        order_data = data["result"].get("orderData", {})
        trans_no = order_data.get("transNo")

        if trans_no:
            log_wallet(self.wallet_id, f"Order placed: {trans_no[:16]}...")

        return trans_no

    def place_limit_order(
        self,
        market_id: int,
        token_id: str,
        side: str,
        price: Decimal,
        size_usdt: Decimal
    ) -> Optional[str]:
        """
        Place a limit order.

        Args:
            market_id: The market/topic ID
            token_id: YES or NO token ID
            side: "BUY" or "SELL"
            price: Price per share (0-1 range, e.g., 0.55 for 55¢)
            size_usdt: Order size in USDT

        Returns:
            Order ID (transNo) if successful, None otherwise
        """
        if not self._proxy_wallet:
            self.logger.error("No proxy wallet - cannot place orders")
            return None

        # Determine side value (0 = BUY, 1 = SELL)
        side_value = 0 if side.upper() == "BUY" else 1

        # Use truncation (not rounding) for price - matches reference bot's round_cut
        # This avoids floating-point rounding issues
        price_float = float(price)
        price_truncated = int(price_float * 1000) / 1000  # Truncate to 3 decimals
        price = Decimal(str(price_truncated))

        # Calculate amounts based on side (matching reference bot logic exactly)
        # Reference bot uses: taker_amount = float(round_cut(amount / price, 2))
        #                    amount = float(Decimal(str(taker_amount)) * Decimal(str(price)))
        amount = float(size_usdt)

        if side_value == 0:  # BUY
            # taker_amount = tokens to receive = USDT / price (truncated to 2 decimals)
            taker_amount_float = int((amount / price_truncated) * 100) / 100
            # maker_amount = USDT to pay (recalculated to match exactly)
            maker_amount_float = float(Decimal(str(taker_amount_float)) * Decimal(str(price_truncated)))

            taker_amount = Decimal(str(taker_amount_float))
            maker_amount = Decimal(str(maker_amount_float))
        else:  # SELL
            # For sell: maker_amount = tokens, taker_amount = USDT received
            maker_amount_float = int((amount / price_truncated) * 100) / 100
            taker_amount_float = float(Decimal(str(maker_amount_float)) * Decimal(str(price_truncated)))

            maker_amount = Decimal(str(maker_amount_float))
            taker_amount = Decimal(str(taker_amount_float))

        log_wallet(
            self.wallet_id,
            f"Placing LIMIT {side.upper()} order: ${size_usdt} @ {float(price)*100:.1f}¢"
        )

        # Build typed data
        typed_data = self._build_order_typed_data(
            token_id=token_id,
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            side=side_value
        )

        # Sign the order
        signature = self._sign_typed_data(typed_data)

        # Submit to API
        return self._submit_order(
            typed_data=typed_data,
            signature=signature,
            market_id=market_id,
            price=price,
            is_market_order=False
        )

    def place_market_order(
        self,
        market_id: int,
        token_id: str,
        side: str,
        size_usdt: Decimal,
        current_price: Decimal
    ) -> Optional[str]:
        """
        Place a market order.

        Market orders execute immediately at the best available price.

        Args:
            market_id: The market/topic ID
            token_id: YES or NO token ID
            side: "BUY" or "SELL"
            size_usdt: Order size in USDT
            current_price: Current market price (used for taker_amount calculation)

        Returns:
            Order ID (transNo) if successful, None otherwise
        """
        if not self._proxy_wallet:
            self.logger.error("No proxy wallet - cannot place orders")
            return None

        # Determine side value (0 = BUY, 1 = SELL)
        side_value = 0 if side.upper() == "BUY" else 1

        # Calculate amounts (market order doesn't need precise calculation, but API expects it)
        if side_value == 0:  # BUY
            maker_amount = size_usdt
            taker_amount = Decimal("0")  # Market orders set taker to 0
        else:  # SELL
            maker_amount = size_usdt / current_price
            taker_amount = size_usdt

        log_wallet(
            self.wallet_id,
            f"Placing MARKET {side.upper()} order: ${size_usdt}"
        )

        # Build typed data
        typed_data = self._build_order_typed_data(
            token_id=token_id,
            maker_amount=maker_amount,
            taker_amount=taker_amount,
            side=side_value
        )

        # Sign the order
        signature = self._sign_typed_data(typed_data)

        # Submit to API
        return self._submit_order(
            typed_data=typed_data,
            signature=signature,
            market_id=market_id,
            price=Decimal("0"),
            is_market_order=True
        )


# ===========================================
# Multi-Wallet Manager
# ===========================================

class OpinionClientManager:
    """Manages multiple Opinion clients (one per wallet)."""

    def __init__(self, wallets: List[WalletConfig], network: NetworkConfig):
        self.network = network
        self.clients: Dict[int, OpinionClient] = {}
        self.logger = get_logger()

        for wallet in wallets:
            self.clients[wallet.wallet_id] = OpinionClient(wallet, network)

    def connect_all(self) -> Tuple[int, int]:
        """Connect all wallets. Returns (success_count, failed_count)."""
        success = 0
        failed = 0

        for wallet_id, client in self.clients.items():
            if client.connect():
                success += 1
            else:
                failed += 1

        self.logger.info(f"Connected {success}/{success + failed} wallets")
        return success, failed

    def disconnect_all(self):
        """Disconnect all wallets."""
        for client in self.clients.values():
            client.disconnect()

    def get_client(self, wallet_id: int) -> Optional[OpinionClient]:
        return self.clients.get(wallet_id)

    def get_all_clients(self) -> List[OpinionClient]:
        return list(self.clients.values())

    def get_connected_clients(self) -> List[OpinionClient]:
        return [c for c in self.clients.values() if c.is_connected]
