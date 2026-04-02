from __future__ import annotations

import unittest

from engine.operation_state import ForegroundOperationRegistry, OperationConflictError


class ForegroundOperationRegistryTests(unittest.TestCase):
    def test_begin_marks_blocking_snapshot(self) -> None:
        registry = ForegroundOperationRegistry()

        snapshot = registry.begin("realtime", "preparing", "Preparing live route.")

        self.assertEqual(snapshot["mode"], "realtime")
        self.assertEqual(snapshot["phase"], "preparing")
        self.assertTrue(snapshot["blocking"])
        self.assertIsNotNone(snapshot["startedAt"])
        self.assertIsNone(snapshot["lastFailure"])

    def test_conflict_raises_with_active_mode_label(self) -> None:
        registry = ForegroundOperationRegistry()
        registry.begin("realtime", "running", "Live voice conversion is active.")

        with self.assertRaises(OperationConflictError) as context:
            registry.ensure_available("single convert")

        self.assertIn("single convert is unavailable", str(context.exception))
        self.assertIn("live voice conversion", str(context.exception))

    def test_fail_is_non_blocking_and_preserves_failure_reason(self) -> None:
        registry = ForegroundOperationRegistry()
        registry.begin("realtime", "preparing", "Preparing live route.")

        snapshot = registry.fail("realtime", "Live failed.", "Input device not available.")

        self.assertEqual(snapshot["mode"], "realtime")
        self.assertEqual(snapshot["phase"], "failed")
        self.assertFalse(snapshot["blocking"])
        self.assertEqual(snapshot["lastFailure"], "Input device not available.")

    def test_snapshot_surfaces_runtime_callback_error_for_realtime(self) -> None:
        registry = ForegroundOperationRegistry()
        registry.begin("realtime", "running", "Live voice conversion is active.")

        snapshot = registry.snapshot(realtime_last_error="callback error")

        self.assertEqual(snapshot["lastFailure"], "callback error")

    def test_clear_restores_idle_state(self) -> None:
        registry = ForegroundOperationRegistry()
        registry.begin("single", "running", "Single convert running.")

        snapshot = registry.clear()

        self.assertEqual(snapshot["mode"], "idle")
        self.assertEqual(snapshot["phase"], "idle")
        self.assertFalse(snapshot["blocking"])
        self.assertEqual(snapshot["message"], "Engine is idle.")


if __name__ == "__main__":
    unittest.main()
