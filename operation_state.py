from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import TypedDict


class OperationSnapshotDict(TypedDict):
    mode: str
    phase: str
    message: str
    blocking: bool
    startedAt: str | None
    lastFailure: str | None


class OperationConflictError(RuntimeError):
    """Raised when a foreground operation is requested while another blocking one owns the engine."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_mode_label(mode: str) -> str:
    return {
        "idle": "the engine",
        "realtime": "live voice conversion",
        "single": "single convert",
        "batch": "batch convert",
        "text": "text audio generation",
    }.get(mode, mode)


@dataclass
class ForegroundOperationState:
    mode: str = "idle"
    phase: str = "idle"
    message: str = "Engine is idle."
    blocking: bool = False
    started_at: str | None = None
    last_failure: str | None = None


class ForegroundOperationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = ForegroundOperationState()

    def snapshot(self, realtime_last_error: str | None = None) -> OperationSnapshotDict:
        with self._lock:
            last_failure = self._state.last_failure
            if self._state.mode == "realtime" and realtime_last_error:
                last_failure = realtime_last_error
            return {
                "mode": self._state.mode,
                "phase": self._state.phase,
                "message": self._state.message,
                "blocking": self._state.blocking,
                "startedAt": self._state.started_at,
                "lastFailure": last_failure,
            }

    def begin(self, mode: str, phase: str, message: str, blocking: bool = True) -> OperationSnapshotDict:
        with self._lock:
            started_at = self._state.started_at if self._state.mode == mode and self._state.started_at else _utc_now_iso()
            self._state = ForegroundOperationState(
                mode=mode,
                phase=phase,
                message=message,
                blocking=blocking,
                started_at=started_at,
                last_failure=None,
            )
            return self.snapshot()

    def fail(self, mode: str, message: str, last_failure: str) -> OperationSnapshotDict:
        with self._lock:
            started_at = self._state.started_at if self._state.mode == mode and self._state.started_at else _utc_now_iso()
            self._state = ForegroundOperationState(
                mode=mode,
                phase="failed",
                message=message,
                blocking=False,
                started_at=started_at,
                last_failure=last_failure,
            )
            return self.snapshot()

    def clear(self) -> OperationSnapshotDict:
        with self._lock:
            self._state = ForegroundOperationState()
            return self.snapshot()

    def ensure_available(self, requester_label: str) -> None:
        with self._lock:
            if not self._state.blocking:
                return
            active_label = _display_mode_label(self._state.mode)
            raise OperationConflictError(f"{requester_label} is unavailable while {active_label} is active.")

    def is_mode_active(self, mode: str) -> bool:
        with self._lock:
            return self._state.mode == mode and self._state.blocking
