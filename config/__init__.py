# ============================================================
# config/__init__.py
# ============================================================
# WHY THIS FILE EXISTS:
# Makes the `config/` directory a Python package so we can do:
#   from config.settings import get_settings
#
# We also re-export the key objects here for convenience,
# so users can do: from config import get_settings
# ============================================================

from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
