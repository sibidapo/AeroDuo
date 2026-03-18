import importlib
import importlib.util
import os
import sys


def disable_broken_flash_attn():
    """Disable FlashAttention if an incompatible binary is installed."""
    if os.environ.get("FLASH_ATTN_DISABLE") is not None:
        return

    if importlib.util.find_spec("flash_attn") is None:
        return

    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        message = str(exc)
        if "flash_attn_2_cuda" not in message and "undefined symbol" not in message:
            return

        os.environ["FLASH_ATTN_DISABLE"] = "1"
        for module_name in list(sys.modules):
            if module_name == "flash_attn" or module_name.startswith("flash_attn."):
                sys.modules.pop(module_name, None)
