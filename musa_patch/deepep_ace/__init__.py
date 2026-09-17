import os

from . import fused_a2a_ace
from . import token_dispatcher


_enable_ace_wgrad = os.getenv("ENABLE_ACE_WGRAD_OVERLAP", "0")
if _enable_ace_wgrad not in ("0", "1"):
    raise ValueError("ENABLE_ACE_WGRAD_OVERLAP must be 0 or 1")
if _enable_ace_wgrad == "1":
    from . import ace_wgrad
