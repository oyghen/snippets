__all__ = ("source_code", "pipe", "retry_call", "retry")

import functools
import inspect
import logging
import time
from collections.abc import Callable, Iterable
from typing import Any, Literal, TypeVar, get_args

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryError(RuntimeError):
    """Raised when all retry attempts for an operation are exhausted."""

    pass


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


def pipe(value: T, functions: Iterable[Callable[[Any], Any]]) -> Any:
    """Return the result of applying a sequence of functions to the initial value."""
    result: Any = value
    for function in functions:
        result = function(result)
    return result


def retry_call(
    func: Callable[..., T],
    attempts: int = 3,
    delay: int | float = 1,
    kind: Literal["fixed", "exponential"] = "exponential",
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    *args: Any,
    **kwargs: Any,
) -> T:
    """Return function result, retrying on specified exceptions with delay."""
    if attempts < 1:
        raise ValueError(f"invalid value {attempts!r}; expected >= 1")
    if delay < 0:
        raise ValueError(f"invalid value {delay!r}; expected >= 0")
    choices = get_args(retry_call.__annotations__["kind"])
    if kind not in choices:
        raise ValueError(f"invalid kind {kind!r}: expected one of {choices}")

    func_name = getattr(func, "__name__", repr(func))
    for attempt in range(attempts):
        try:
            logger.info("calling %s (attempt %d/%d)", func_name, attempt + 1, attempts)
            return func(*args, **kwargs)
        except exceptions as exc:
            is_last = attempt == attempts - 1
            sleep_for = delay * (2**attempt) if kind == "exponential" else delay

            if is_last:
                logger.exception(
                    "final attempt %d/%d for %s failed: raising exception",
                    attempt + 1,
                    attempts,
                    func_name,
                )
                message = (
                    f"retries exhausted for {func_name} after {attempts} attempt(s); ",
                    f"last error: {exc!r}",
                )
                raise RetryError(message) from exc

            logger.warning(
                "attempt %d/%d for %s failed with %r; retrying in %s seconds",
                attempt + 1,
                attempts,
                func_name,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)


def retry(
    attempts: int = 3,
    delay: int | float = 1,
    kind: Literal["fixed", "exponential"] = "exponential",
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries the wrapped function on specified exceptions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        """Return wrapped function that retries on the configured exceptions."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_call(func, attempts, delay, kind, exceptions, *args, **kwargs)

        return wrapper

    return decorator
