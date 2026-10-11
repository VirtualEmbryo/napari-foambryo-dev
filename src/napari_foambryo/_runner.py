"""Run one library call in a child process that Cancel can stop, and that returns its memory when it ends.

A library call cannot be interrupted inside a thread, and a reconstruction can hold tens of GiB; so a
reconstruction, or a seed ensemble, runs in a child process (`spawn` start method), in a process group of its own:
Cancel terminates the whole group (SIGTERM, then SIGKILL after a grace period), the ensemble's own worker
processes included. The child returns its result by pickling, with the warnings it raised (each once, with its
count, `foambryo.results.WarningCollector`). The parent polls the child's resident memory (and its children's) and
reports the peak. Nothing partial is ever returned: a cancelled or failed run returns nothing.

The call is named by `"module:function"`, so that the child imports exactly the library function the command and
plain Python call: nothing scientific is computed here.
"""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing
import os
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ChildCancelledError", "ChildFailedError", "ChildResult", "run_in_child"]

#: Seconds between SIGTERM and SIGKILL when a run is cancelled.
TERMINATE_GRACE_SECONDS = 5.0


class ChildCancelledError(Exception):
    """The run was cancelled; nothing was returned."""


class ChildFailedError(Exception):
    """The library refused, or the child process ended without a result.

    Attributes:
        kind: the exception's class name in the child (`"ValueError"`, ...), or `"ProcessEnded"`.
        message: its message.
        child_traceback: the child's traceback, as text (empty when the process ended without one).
        warnings_summary: the warnings raised before the failure.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        child_traceback: str = "",
        warnings_summary: list | None = None,
    ) -> None:
        """Hold what the child said."""
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.message = message
        self.child_traceback = child_traceback
        self.warnings_summary = warnings_summary or []


@dataclass
class ChildResult:
    """What a child run returned.

    Attributes:
        value: the library function's return value.
        warnings_summary: the warnings it raised, each once with its count.
        peak_rss_bytes: the largest resident memory of the child and its own children, as polled.
        seconds: the wall time of the run, start of the child included.
    """

    value: Any
    warnings_summary: list = field(default_factory=list)
    peak_rss_bytes: int = 0
    seconds: float = 0.0


def _resolve(target: str) -> Callable[..., Any]:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _child_main(connection: Any, target: str, kwargs: dict, with_progress: bool) -> None:  # noqa: ANN401
    """The child: a process group of its own, then the call, its warnings counted, the outcome sent back."""
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    from foambryo.results import WarningCollector

    collector = WarningCollector()
    try:
        function = _resolve(target)
        if with_progress:
            kwargs = {**kwargs, "progress": lambda *args: connection.send(("progress", args))}
        with collector:
            value = function(**kwargs)
        connection.send(("ok", value, collector.summary))
    except BaseException as error:  # everything is reported to the parent, which decides
        connection.send(("error", type(error).__name__, str(error), traceback.format_exc(), collector.summary))
    finally:
        connection.close()


def _rss_of_tree(process: Any) -> int:  # noqa: ANN401 - a psutil.Process
    import psutil

    try:
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            with contextlib.suppress(psutil.Error):
                total += child.memory_info().rss
    except psutil.Error:
        return 0
    return total


def _stop_group(process: multiprocessing.process.BaseProcess) -> None:
    """SIGTERM to the child's process group, then SIGKILL after the grace period."""
    pid = process.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, AttributeError):
        process.terminate()  # the child had not made its group yet
    process.join(TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            process.kill()
        process.join()


def run_in_child(
    target: str,
    kwargs: dict[str, Any],
    *,
    should_cancel: Callable[[], bool] = lambda: False,
    on_progress: Callable[..., None] | None = None,
    on_memory: Callable[[int], None] | None = None,
    poll_seconds: float = 0.2,
) -> ChildResult:
    """Call `target(**kwargs)` in a child process and return its result.

    Args:
        target: `"module:function"`, for example `"foambryo.io:reconstruct_mask"`.
        kwargs: the call's keyword arguments, sent by pickling.
        should_cancel: polled; when it returns `True` the child's process group is stopped.
        on_progress: when given, the call receives a `progress` callback whose arguments arrive here.
        on_memory: called at each poll with the resident memory of the child and its children, in bytes.
        poll_seconds: the polling interval.

    Returns:
        ChildResult: the value, the warnings, the peak memory, the time.

    Raises:
        ChildCancelledError: when `should_cancel` returned `True`.
        ChildFailedError: when the call raised, or the child ended without a result (for example killed by the
            system for lack of memory).
    """
    import psutil

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child_main, args=(send, target, kwargs, on_progress is not None))
    started = time.perf_counter()
    process.start()
    send.close()
    watched = psutil.Process(process.pid)
    peak = 0
    try:
        while True:
            if should_cancel():
                _stop_group(process)
                raise ChildCancelledError
            if receive.poll(poll_seconds):
                try:
                    message = receive.recv()
                except EOFError:
                    message = None
                if message is not None and message[0] == "progress":
                    if on_progress is not None:
                        on_progress(*message[1])
                    continue
                process.join()
                if message is None:
                    raise ChildFailedError(
                        "ProcessEnded", f"the child process ended without a result (exit code {process.exitcode})"
                    )
                if message[0] == "ok":
                    return ChildResult(message[1], message[2], peak, time.perf_counter() - started)
                raise ChildFailedError(message[1], message[2], message[3], message[4])
            rss = _rss_of_tree(watched)
            peak = max(peak, rss)
            if on_memory is not None and rss:
                on_memory(rss)
            if not process.is_alive() and not receive.poll(0):
                process.join()
                raise ChildFailedError(
                    "ProcessEnded",
                    f"the child process ended without a result (exit code "
                    f"{process.exitcode}; a negative code is the signal that "
                    "stopped it, for example 9 when the system ran out of memory)",
                )
    finally:
        receive.close()
        if process.is_alive():
            _stop_group(process)
