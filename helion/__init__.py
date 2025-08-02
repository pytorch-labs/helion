from __future__ import annotations

from triton import cdiv
from triton import next_power_of_2 as _triton_next_power_of_2

from . import _logging
from . import exc
from . import language
from . import runtime
from .runtime import Config
from .runtime import Kernel
from .runtime import kernel
from .runtime import kernel as jit  # alias
from .runtime.settings import RefMode
from .runtime.settings import Settings
from .runtime.settings import set_default_settings


def next_power_of_2(n: int) -> int | object:
    """Compute the next power of 2 for a given number.

    In ref mode, returns a DimSize object that tracks both
    logical and physical sizes. In normal mode, returns an int.
    """
    from .runtime.ref_mode import is_in_ref_mode_context

    if is_in_ref_mode_context():
        from .language.ref_tensor import DimSize

        return DimSize(n)
    return _triton_next_power_of_2(n)


__all__ = [
    "Config",
    "Kernel",
    "RefMode",
    "Settings",
    "cdiv",
    "exc",
    "jit",
    "kernel",
    "language",
    "next_power_of_2",
    "runtime",
    "set_default_settings",
]

_logging.init_logs()
