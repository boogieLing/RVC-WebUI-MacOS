from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from engine.realtime_vc import RealtimeVCController


def _make_query_devices():
    devices = [
        {
            "name": "Mic",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "index": 11,
            "default_samplerate": 48000,
        },
        {
            "name": "Speaker",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "index": 22,
            "default_samplerate": 48000,
        },
    ]
    by_index = {device["index"]: dict(device) for device in devices}

    def query_devices(device=None, kind=None):
        del kind
        if device is None:
            return [dict(item) for item in devices]
        return dict(by_index[device])

    return query_devices


class RealtimeVCControllerTests(unittest.TestCase):
    def _make_controller(self, fake_default: SimpleNamespace) -> RealtimeVCController:
        app_config = SimpleNamespace(device="cpu", use_jit=False, is_half=False, dml=False)
        logger = SimpleNamespace(exception=lambda *args, **kwargs: None)

        with patch("engine.realtime_vc.sd._terminate"), \
             patch("engine.realtime_vc.sd._initialize"), \
             patch("engine.realtime_vc.sd.query_devices", side_effect=_make_query_devices()), \
             patch("engine.realtime_vc.sd.query_hostapis", return_value=[{"name": "Core Audio", "devices": [0, 1]}]), \
             patch("engine.realtime_vc.sd.default", fake_default):
            return RealtimeVCController(app_config, logger)

    def test_configure_does_not_bind_devices_before_live_start(self) -> None:
        fake_default = SimpleNamespace(device=[-1, -1], reset=MagicMock())
        controller = self._make_controller(fake_default)
        controller.set_devices = MagicMock()

        with patch("engine.realtime_vc.sd.query_devices", side_effect=_make_query_devices()):
            result = controller.configure(
                {
                    "inputDevice": "Mic",
                    "outputDevice": "Speaker",
                    "hostapi": "Core Audio",
                },
                allow_restart=True,
            )

        controller.set_devices.assert_not_called()
        self.assertEqual(result["status"]["selectedInputDevice"], "Mic")
        self.assertEqual(result["status"]["selectedOutputDevice"], "Speaker")
        self.assertEqual(result["status"]["sampleRate"], 48000)
        self.assertEqual(result["status"]["channels"], 1)

    def test_stop_stream_resets_default_device_binding(self) -> None:
        fake_default = SimpleNamespace(device=[11, 22], reset=MagicMock())
        controller = self._make_controller(fake_default)
        controller.stream = SimpleNamespace(abort=MagicMock(), close=MagicMock())
        controller.running = True

        controller.stop_stream()

        controller.stream = None
        fake_default.reset.assert_called_once()
        self.assertFalse(controller.running)


if __name__ == "__main__":
    unittest.main()
