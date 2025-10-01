from __future__ import annotations

import logging
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "evalguard") -> logging.Logger:
    global _LOGGER
    if _LOGGER is None:
        console = Console()
        handler = RichHandler(console=console, show_path=False, markup=True, rich_tracebacks=True)
        logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(message)s")
        _LOGGER = logging.getLogger(name)
    return logging.getLogger(name)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
