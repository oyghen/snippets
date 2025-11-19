__all__ = ["source_code"]

import inspect
from typing import Any


def source_code(obj: Any, /) -> str:
    """Return the text of the source code for an object.

    Examples
    --------
    >>> from snippy import core
    >>> def greet():
    ...     return "Hello, World!"
    ...
    >>> print(core.source_code(greet))
    def greet():
        return "Hello, World!"
    <BLANKLINE>
    """
    return inspect.getsource(obj)
