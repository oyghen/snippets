__all__ = (
    "source_code",
    "pipe",
    "retry_call",
    "retry",
    "confirm",
    "get_last_commit_hash",
    "get_installed_package_count",
    "iter_lines",
    "terminal_width",
)

import functools
import inspect
import logging
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from importlib import metadata
from pathlib import Path
from shutil import get_terminal_size
from typing import Any, Literal, TypeAlias, TypeVar, get_args

logger = logging.getLogger(__name__)

T = TypeVar("T")
PathLikeStr: TypeAlias = str | Path | os.PathLike[str]


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


def confirm(prompt: str, default: str | None = None) -> bool:
    """Return True if the user confirms ('yes'); repeats until valid input."""
    if default not in (None, "yes", "no"):
        raise ValueError(f"invalid {default=!r}; expected None, 'yes', or 'no'")

    true_tokens = frozenset(("y", "yes", "t", "true", "on", "1"))
    false_tokens = frozenset(("n", "no", "f", "false", "off", "0"))
    prompt_map = {None: "[y/n]", "yes": "[Y/n]", "no": "[y/N]"}
    suffix = prompt_map[default]

    while True:
        reply = input(f"{prompt} {suffix} ").strip().lower()

        if not reply:
            if default is not None:
                return default == "yes"
            print("Please respond with 'yes' or 'no'.")
            continue

        if reply in true_tokens:
            return True
        if reply in false_tokens:
            return False

        print("Please respond with 'yes' or 'no'.")


def get_last_commit_hash(max_length: int | None = None) -> str:
    """Return the full or truncated commit hash of the current branch."""
    if max_length is not None:
        if not isinstance(max_length, int):
            raise TypeError(
                f"unsupported type {type(max_length).__name__!r}; expected int or None"
            )
        if max_length < 1:
            raise ValueError(f"invalid value {max_length!r}; expected >= 1")

    commit = subprocess.run(
        shlex.split("git rev-parse HEAD"), capture_output=True, text=True, check=True
    )
    if not commit:
        raise RuntimeError("git returned an empty commit hash")

    return commit if max_length is None else commit[:max_length]


def get_installed_package_count() -> int:
    """Return the number of installed packages for the current Python environment."""
    try:
        count = sum(1 for _ in metadata.distributions())
    except Exception:
        count = 0

    # fallback: use the same interpreter's pip to get a reliable package list
    if count < 10:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                check=True,
                capture_output=True,
                text=True,
            )
            # ignore blank lines and count non-empty entries
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            return len(lines)
        except (subprocess.SubprocessError, OSError):
            # if pip fails, return what metadata provided (possibly 0)
            return int(count)

    return int(count)


def iter_lines(
    filepath: PathLikeStr,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> Iterator[str]:
    """Return an iterator over text lines from filepath."""
    path = os.fspath(filepath)
    with open(path, encoding=encoding, errors=errors, newline=newline) as fh:
        yield from fh


def terminal_width(default: int = 79) -> int:
    """Return the current terminal width or a fallback value."""
    try:
        return get_terminal_size().columns
    except OSError:
        return default
