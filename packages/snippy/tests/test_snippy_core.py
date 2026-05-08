import builtins
import logging
import time
import types
from collections.abc import Iterable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, NamedTuple
from unittest.mock import Mock

import pytest
from snippy import core


def test_source_code():
    def greet():
        return "Hello, World!"

    actual = core.source_code(greet)
    expected = '    def greet():\n        return "Hello, World!"\n'
    assert actual == expected


class TestPipe:
    def test_pipe_single_function(self):
        assert core.pipe(2, [lambda x: x + 1]) == 3

    def test_pipe_multiple_functions(self):
        assert core.pipe(2, [lambda x: x + 1, lambda x: x * 3]) == 9

    def test_pipe_heterogeneous_types(self):
        def to_str(x: int) -> str:
            return f"{x=}"

        def to_len(s: str) -> int:
            return len(s)

        assert core.pipe(5, [to_str, to_len]) == len("x=5")

    def test_pipe_no_functions_returns_input(self):
        assert core.pipe("data", []) == "data"

    @pytest.mark.parametrize(
        "data, functions, expected",
        [
            (3, [lambda x: x * 2], 6),
            ("4", [int, lambda n: n * n, str], "16"),
            ("5", [int, float, lambda x: x > 0], True),
            ("unchanged", [], "unchanged"),
            ("hello", [lambda x: x, lambda x: x], "hello"),
        ],
        ids=[
            "single function",
            "multiple functions",
            "type transformations",
            "empty pipeline",
            "identity pipeline",
        ],
    )
    def test_pipe(self, data: Any, functions: list, expected: Any):
        assert core.pipe(data, functions) == expected

    def test_function_raises_error(self):
        def bad_fn(x: Any) -> Any:
            raise ValueError("Oops!")

        with pytest.raises(ValueError):
            core.pipe(10, [bad_fn])

    def test_pipe_functions_called_in_order(self):
        f1 = Mock(side_effect=lambda x: x + 1)
        f2 = Mock(side_effect=lambda x: x * 2)

        result = core.pipe(3, [f1, f2])

        assert result == 8
        f1.assert_called_once_with(3)
        f2.assert_called_once_with(4)


class TestRetry:
    def test_retry_call_success_first_try(self):
        def func():
            return 42

        assert core.retry_call(func) == 42

    @pytest.mark.parametrize(
        "attempts, fail_times",
        [
            (3, 2),
            (5, 4),
            (1, 0),
        ],
    )
    def test_retry_call_retries_and_succeeds(self, attempts: int, fail_times: int):
        state = SimpleNamespace(count=0)

        def func():
            if state.count < fail_times:
                state.count += 1
                raise ValueError("fail")
            return "success"

        result = core.retry_call(
            func,
            attempts=attempts,
            delay=0,
            exceptions=(ValueError,),
        )
        assert result == "success"
        assert state.count == fail_times

    def test_retry_call_raises_after_max_attempts(self):
        max_attempts = 3
        state = SimpleNamespace(count=0)

        def func():
            state.count += 1
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            core.retry_call(func, max_attempts, delay=0, exceptions=(RuntimeError,))

        assert state.count == 3

    @pytest.mark.parametrize(
        "attempts, fail_times",
        [
            (3, 2),
            (5, 4),
            (1, 0),
        ],
    )
    def test_retry_decorator_retries_and_succeeds(self, attempts: int, fail_times: int):
        state = SimpleNamespace(count=0)

        @core.retry(attempts, delay=0, exceptions=(KeyError,))
        def func():
            if state.count < fail_times:
                state.count += 1
                raise KeyError("fail")
            return "success"

        result = func()
        assert result == "success"
        assert state.count == fail_times

    def test_retry_decorator_raises_after_max_attempts(self):
        max_attempts = 3
        state = SimpleNamespace(count=0)

        @core.retry(max_attempts, delay=0, exceptions=(RuntimeError,))
        def func():
            state.count += 1
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            func()

        assert state.count == 3

    @pytest.mark.parametrize(
        "kind, attempts, delay, expected_sleeps",
        [
            ("fixed", 3, 1, [1, 1]),
            ("exponential", 4, 0.5, [0.5, 1.0, 2.0]),
            ("fixed", 7, 1, [1, 1, 1, 1, 1, 1]),
            ("exponential", 7, 1, [1, 2, 4, 8, 16, 32]),
        ],
    )
    def test_retry_call_sleep_durations(
        self,
        monkeypatch,
        kind: str,
        attempts: int,
        delay: int | float,
        expected_sleeps: list[int | float],
    ):
        state = SimpleNamespace(count=0)
        sleeps = []

        def fake_sleep(duration):
            sleeps.append(duration)

        monkeypatch.setattr(time, "sleep", fake_sleep)

        def func():
            if state.count < attempts - 1:
                state.count += 1
                raise ValueError("fail")
            return "success"

        result = core.retry_call(
            func,
            attempts=attempts,
            delay=delay,
            kind=kind,
            exceptions=(ValueError,),
        )

        assert result == "success"
        assert state.count == attempts - 1
        assert sleeps == expected_sleeps


class TestRetryLogging:
    def test_info_log_on_first_call(self, caplog):
        caplog.set_level(logging.INFO)

        def add_one(x: int):
            return x + 1

        result = core.retry_call(add_one, attempts=3, delay=0, x=1)
        assert result == 2

        assert len(caplog.records) == 1
        rec = caplog.records[0]
        msg = rec.getMessage()
        assert rec.levelno == logging.INFO
        assert msg.startswith(f"calling {add_one.__name__} (attempt 1/3)")

    @pytest.mark.parametrize(
        "attempts, fail_times",
        [
            (3, 2),
            (5, 4),
            (1, 0),
        ],
    )
    def test_warning_logged_on_retry(self, caplog, attempts: int, fail_times: int):
        caplog.set_level(logging.WARNING)
        state = SimpleNamespace(count=0)

        def func():
            if state.count < fail_times:
                state.count += 1
                raise RuntimeError("fail")
            return "success"

        result = core.retry_call(func, attempts, delay=0)
        assert result == "success"
        assert state.count == fail_times

        assert len(caplog.records) == fail_times
        for i in range(fail_times):
            rec = caplog.records[i]
            msg = rec.getMessage()
            assert rec.levelno == logging.WARNING
            assert msg.startswith(
                f"attempt {i + 1}/{attempts} for {func.__name__} failed"
            )
            assert "retrying in" in msg
            assert msg.endswith("seconds")

    @pytest.mark.parametrize("attempts", [1, 3, 5])
    def test_exception_logged_on_final_attempt(self, caplog, attempts: int):
        caplog.set_level(logging.ERROR)
        state = SimpleNamespace(count=0)

        def always_fail():
            state.count += 1
            raise RuntimeError("fail")

        with pytest.raises(core.RetryError):
            core.retry_call(always_fail, attempts, delay=0)

        assert state.count == attempts

        assert len(caplog.records) == 1
        rec = caplog.records[0]
        msg = rec.getMessage()
        assert rec.levelno == logging.ERROR
        assert msg.startswith(
            f"final attempt {attempts}/{attempts} for {always_fail.__name__} failed"
        )
        assert msg.endswith("raising exception")


class TestConfirm:
    def test_accepts_yes(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", self.patch_input(["y"]))
        assert core.confirm("Proceed?") is True

    def test_accepts_no(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", self.patch_input(["n"]))
        assert core.confirm("Proceed?") is False

    def test_uses_default_yes_on_empty_reply(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", self.patch_input([""]))
        assert core.confirm("Proceed?", default="yes") is True

    def test_uses_default_no_on_empty_reply(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", self.patch_input([""]))
        assert core.confirm("Proceed?", default="no") is False

    def test_reprompts_on_invalid_then_accepts(self, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", self.patch_input(["maybe", "yes"]))
        result = core.confirm("Proceed?")
        captured = capsys.readouterr()
        assert "Please respond with 'yes' or 'no'." in captured.out
        assert result is True

    def test_whitespace_counts_as_empty_and_reprompts(self, monkeypatch, capsys):
        monkeypatch.setattr(builtins, "input", self.patch_input(["   ", "1"]))
        result = core.confirm("Proceed?")
        captured = capsys.readouterr()
        assert "Please respond with 'yes' or 'no'." in captured.out
        assert result is True  # '1' maps to true tokens

    def test_invalid_default_raises(self):
        with pytest.raises(ValueError):
            core.confirm("Proceed?", default="maybe")

    @staticmethod
    def patch_input(responses: list[str]):
        """Return a stub for builtins.input that yields the given responses."""
        iterator = iter(responses)

        def _input(prompt: str) -> str:
            try:
                return next(iterator)
            except StopIteration as exc:
                raise AssertionError("test provided too few responses") from exc

        return _input


class TestIterLines:
    def test_returns_generator(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("x\n", encoding="utf-8")

            gen = core.iter_lines(p)
            assert isinstance(gen, types.GeneratorType)
            gen.close()

    def test_reads_lines_preserving_newlines(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("a\nb\n", encoding="utf-8")

            assert list(core.iter_lines(p)) == ["a\n", "b\n"]

    def test_accepts_str_path(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("1\n2\n", encoding="utf-8")

            assert list(core.iter_lines(str(p))) == ["1\n", "2\n"]

    def test_encoding_and_errors(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("café\n", encoding="utf-8")

            result = list(core.iter_lines(p, encoding="ascii", errors="replace"))
            assert result == ["caf\ufffd\ufffd\n"]

    def test_exhaustion_closes_generator(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("line\n", encoding="utf-8")

            gen = core.iter_lines(p)
            assert list(gen) == ["line\n"]
            assert getattr(gen, "gi_frame", None) is None

    def test_explicit_close_closes_file(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("line\n", encoding="utf-8")

            gen = core.iter_lines(p)
            assert next(gen) == "line\n"
            gen.close()
            assert getattr(gen, "gi_frame", None) is None

    def test_pipe_lines(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.txt"
            p.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

            def strip_lines(lines: Iterable[str]) -> Iterator[str]:
                """Return an iterator yielding stripped lines."""
                return (s.strip() for s in lines)

            def filter_len_gt_3(lines: Iterable[str]) -> Iterator[str]:
                """Return an iterator filtering lines with length > 3."""
                return (s for s in lines if len(s) > 3)

            funcs = [strip_lines, filter_len_gt_3, tuple]
            result = core.pipe(core.iter_lines(p), funcs)

            assert result == ("three", "four")


class TestTerminalWidth:
    class TerminalSize(NamedTuple):
        columns: int
        lines: int

    def test_from_shutil(self, monkeypatch):
        # simulate a terminal width of 100
        monkeypatch.setattr(
            core,
            "get_terminal_size",
            lambda: self.TerminalSize(columns=100, lines=24),
        )
        assert core.terminal_width(default=79) == 100

    def test_fallback(self, monkeypatch):
        def raise_oserror():
            raise OSError("no tty")

        monkeypatch.setattr(core, "get_terminal_size", raise_oserror)
        assert core.terminal_width(default=79) == 79
