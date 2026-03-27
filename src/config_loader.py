"""
Configuration loader for Opinion Trade Bot.

This module handles loading:
- config.yaml (bot settings)
- wallets.txt (private keys)
- proxies.txt (optional proxy settings)
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .utils.logging_utils import get_logger


# ===========================================
# Data Classes (simple containers for data)
# ===========================================

@dataclass
class WalletConfig:
    """
    Holds information for one wallet.

    Attributes:
        wallet_id: The wallet number (1, 2, 3, etc.)
        private_key: The private key (64 hex chars, with or without 0x)
        address: The wallet address (derived from private key, filled later)
        proxy: Optional proxy string for this wallet
    """
    wallet_id: int
    private_key: str
    address: str = ""  # Will be derived from private key when client connects
    proxy: Optional[str] = None

    def has_proxy(self) -> bool:
        """Check if this wallet has a proxy configured."""
        return self.proxy is not None and len(self.proxy) > 0

    def get_proxy_dict(self) -> Optional[Dict[str, str]]:
        """
        Convert proxy string to requests-compatible dict.

        Returns:
            Dict like {"http": "...", "https": "..."} or None if no proxy
        """
        if not self.has_proxy():
            return None

        proxy_str = self.proxy

        # Parse proxy string: ip:port or ip:port:user:pass
        parts = proxy_str.split(':')

        if len(parts) == 2:
            # ip:port (no auth)
            host, port = parts
            proxy_url = f"http://{host}:{port}"
        elif len(parts) == 4:
            # ip:port:user:pass
            host, port, user, password = parts
            proxy_url = f"http://{user}:{password}@{host}:{port}"
        else:
            get_logger().warning(f"Invalid proxy format: {proxy_str}")
            return None

        return {
            "http": proxy_url,
            "https": proxy_url
        }


@dataclass
class NetworkConfig:
    """Network/blockchain settings."""
    chain_id: int = 56
    rpc_url: str = "https://bsc-dataseed.binance.org"
    host: str = "https://proxy.opinion.trade:8443"


@dataclass
class MarketFilters:
    """Filters for market discovery."""
    min_24h_volume: float = 100
    min_open_interest: float = 0
    top_n_by_volume: int = 10
    top_n_by_low_oi: int = 10


@dataclass
class TradingConfig:
    """Trading behavior settings."""
    poll_interval_seconds: float = 2
    default_tick_offset: int = 1
    default_order_size_usdt: float = 10
    max_open_orders: int = 5


@dataclass
class OrderSizeConfig:
    """Order sizing settings."""
    mode: str = "fixed"  # "fixed" or "percentage"
    fixed_amount: float = 10
    percentage: float = 2


@dataclass
class LoggingConfig:
    """Logging settings."""
    level: str = "INFO"
    log_to_file: bool = True
    log_file_prefix: str = "bot"


@dataclass
class BotConfig:
    """
    Master configuration object that holds everything.
    """
    network: NetworkConfig = field(default_factory=NetworkConfig)
    market_filters: MarketFilters = field(default_factory=MarketFilters)
    trading: TradingConfig = field(default_factory=TradingConfig)
    order_size: OrderSizeConfig = field(default_factory=OrderSizeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wallets: List[WalletConfig] = field(default_factory=list)

    def get_wallet(self, wallet_id: int) -> Optional[WalletConfig]:
        """Get a wallet by its ID."""
        for w in self.wallets:
            if w.wallet_id == wallet_id:
                return w
        return None

    def get_all_wallet_ids(self) -> List[int]:
        """Get list of all wallet IDs."""
        return [w.wallet_id for w in self.wallets]


# ===========================================
# Loading Functions
# ===========================================

def load_wallets(wallets_file: str = "input_data/wallets.txt") -> List[WalletConfig]:
    """
    Load wallets from the wallets.txt file.

    Format: wallet_number, private_key
    Example: 1, 0xabcdef...

    Args:
        wallets_file: Path to the wallets file

    Returns:
        List of WalletConfig objects
    """
    wallets = []
    file_path = Path(wallets_file)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Wallets file not found: {wallets_file}\n"
            f"Please copy wallets.txt.example to wallets.txt and add your private keys."
        )

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse: number, private_key
            if ',' not in line:
                get_logger().warning(f"Skipping invalid line {line_num} in wallets.txt: no comma found")
                continue

            parts = line.split(',', 1)  # Split only on first comma
            if len(parts) != 2:
                get_logger().warning(f"Skipping invalid line {line_num} in wallets.txt")
                continue

            try:
                wallet_id = int(parts[0].strip())
                private_key = parts[1].strip()

                # Validate private key format (should be 64 hex chars, optionally with 0x)
                key_clean = private_key.lower()
                if key_clean.startswith('0x'):
                    key_clean = key_clean[2:]

                if len(key_clean) != 64 or not all(c in '0123456789abcdef' for c in key_clean):
                    get_logger().warning(
                        f"Line {line_num}: Private key doesn't look valid (should be 64 hex chars)"
                    )
                    continue

                # Normalize: always store with 0x prefix
                if not private_key.startswith('0x'):
                    private_key = '0x' + private_key

                wallets.append(WalletConfig(
                    wallet_id=wallet_id,
                    private_key=private_key
                ))

            except ValueError as e:
                get_logger().warning(f"Skipping invalid line {line_num} in wallets.txt: {e}")
                continue

    if not wallets:
        raise ValueError("No valid wallets found in wallets.txt")

    get_logger().info(f"Loaded {len(wallets)} wallet(s) from {wallets_file}")
    return wallets


def load_proxies(proxies_file: str = "input_data/proxies.txt") -> Dict[int, str]:
    """
    Load proxies from the proxies.txt file.

    Format: wallet_number, proxy_string
    Example: 1, 192.168.1.100:8080:user:pass

    Args:
        proxies_file: Path to the proxies file

    Returns:
        Dict mapping wallet_id -> proxy_string
    """
    proxies = {}
    file_path = Path(proxies_file)

    # Proxies are optional - if file doesn't exist, return empty dict
    if not file_path.exists():
        get_logger().info(f"No proxies file found at {proxies_file} - running without proxies")
        return proxies

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse: number, proxy_string
            if ',' not in line:
                get_logger().warning(f"Skipping invalid line {line_num} in proxies.txt: no comma found")
                continue

            parts = line.split(',', 1)
            if len(parts) != 2:
                get_logger().warning(f"Skipping invalid line {line_num} in proxies.txt")
                continue

            try:
                wallet_id = int(parts[0].strip())
                proxy_string = parts[1].strip()

                # Basic validation of proxy format
                if ':' not in proxy_string:
                    get_logger().warning(f"Line {line_num}: Invalid proxy format (no port found)")
                    continue

                proxies[wallet_id] = proxy_string

            except ValueError as e:
                get_logger().warning(f"Skipping invalid line {line_num} in proxies.txt: {e}")
                continue

    if proxies:
        get_logger().info(f"Loaded {len(proxies)} proxy configuration(s) from {proxies_file}")

    return proxies


def load_config(config_file: str = "config.yaml") -> BotConfig:
    """
    Load the complete bot configuration from all sources.

    This function:
    1. Loads config.yaml for bot settings
    2. Loads wallets.txt for private keys
    3. Loads proxies.txt (optional) and assigns proxies to wallets

    Args:
        config_file: Path to the YAML config file

    Returns:
        Complete BotConfig object
    """
    config = BotConfig()

    # Load YAML config
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f) or {}

        # Parse network settings
        if 'network' in yaml_data:
            net = yaml_data['network']
            config.network = NetworkConfig(
                chain_id=net.get('chain_id', 56),
                rpc_url=net.get('rpc_url', "https://bsc-dataseed.binance.org"),
                host=net.get('host', "https://proxy.opinion.trade:8443")
            )

        # Parse market filters
        if 'market_filters' in yaml_data:
            mf = yaml_data['market_filters']
            config.market_filters = MarketFilters(
                min_24h_volume=mf.get('min_24h_volume', 100),
                min_open_interest=mf.get('min_open_interest', 0),
                top_n_by_volume=mf.get('top_n_by_volume', 10),
                top_n_by_low_oi=mf.get('top_n_by_low_oi', 10)
            )

        # Parse trading settings
        if 'trading' in yaml_data:
            tr = yaml_data['trading']
            config.trading = TradingConfig(
                poll_interval_seconds=tr.get('poll_interval_seconds', 2),
                default_tick_offset=tr.get('default_tick_offset', 1),
                default_order_size_usdt=tr.get('default_order_size_usdt', 10),
                max_open_orders=tr.get('max_open_orders', 5)
            )

        # Parse order size settings
        if 'order_size' in yaml_data:
            os_cfg = yaml_data['order_size']
            config.order_size = OrderSizeConfig(
                mode=os_cfg.get('mode', 'fixed'),
                fixed_amount=os_cfg.get('fixed_amount', 10),
                percentage=os_cfg.get('percentage', 2)
            )

        # Parse logging settings
        if 'logging' in yaml_data:
            log = yaml_data['logging']
            config.logging = LoggingConfig(
                level=log.get('level', 'INFO'),
                log_to_file=log.get('log_to_file', True),
                log_file_prefix=log.get('log_file_prefix', 'bot')
            )

        get_logger().info(f"Loaded configuration from {config_file}")
    else:
        get_logger().warning(f"Config file not found: {config_file} - using defaults")

    # Load wallets
    config.wallets = load_wallets()

    # Load and assign proxies
    proxies = load_proxies()
    for wallet in config.wallets:
        if wallet.wallet_id in proxies:
            wallet.proxy = proxies[wallet.wallet_id]
            get_logger().debug(f"Wallet #{wallet.wallet_id} assigned proxy: {wallet.proxy}")

    return config


# ===========================================
# Utility Functions
# ===========================================

def print_config_summary(config: BotConfig):
    """
    Print a human-readable summary of the loaded configuration.
    """
    logger = get_logger()

    logger.info("=" * 50)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 50)

    # Network
    logger.info(f"Network: Chain ID {config.network.chain_id}")
    logger.info(f"RPC URL: {config.network.rpc_url}")
    logger.info(f"API Host: {config.network.host}")

    # Wallets
    logger.info(f"Wallets loaded: {len(config.wallets)}")
    for w in config.wallets:
        proxy_status = f"via proxy {w.proxy}" if w.has_proxy() else "direct connection"
        # Only show first/last 4 chars of private key for security
        key_preview = f"{w.private_key[:6]}...{w.private_key[-4:]}"
        logger.info(f"  Wallet #{w.wallet_id}: {key_preview} ({proxy_status})")

    # Trading
    logger.info(f"Poll interval: {config.trading.poll_interval_seconds}s")
    logger.info(f"Default tick offset: {config.trading.default_tick_offset}")
    logger.info(f"Max open orders: {config.trading.max_open_orders}")

    logger.info("=" * 50)
