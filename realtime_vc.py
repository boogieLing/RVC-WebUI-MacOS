from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

import infer.lib.rtrvc as rtrvc
from infer.modules.gui import TorchGate


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


@dataclass
class RealtimeVCConfig:
    model_name: str = ""
    index_path: str = ""
    pitch: int = 0
    formant: float = 0.0
    sr_type: str = "sr_model"
    block_time: float = 0.25
    threhold: int = -60
    crossfade_time: float = 0.05
    extra_time: float = 2.5
    input_noise_reduce: bool = False
    output_noise_reduce: bool = False
    use_pv: bool = False
    rms_mix_rate: float = 0.0
    index_rate: float = 0.0
    n_cpu: int = min(os.cpu_count() or 1, 4)
    f0method: str = "fcpe"
    sg_hostapi: str = ""
    wasapi_exclusive: bool = False
    sg_input_device: str = ""
    sg_output_device: str = ""
    function: str = "vc"
    samplerate: int = 0
    channels: int = 1


class RealtimeVCController:
    def __init__(self, app_config, logger):
        self.config = app_config
        self.logger = logger
        self.gui_config = RealtimeVCConfig()
        self.hostapis: list[str] = []
        self.input_devices: list[str] = []
        self.output_devices: list[str] = []
        self.input_devices_indices: list[int] = []
        self.output_devices_indices: list[int] = []
        self.stream = None
        self.rvc = None
        self.running = False
        self.delay_time_ms = 0
        self.infer_time_ms = 0
        self.last_error: Optional[str] = None
        self.lock = threading.RLock()
        self.update_devices()

    def devices_snapshot(self) -> dict:
        return {
            "hostapis": self.hostapis,
            "selectedHostapi": self.gui_config.sg_hostapi,
            "inputDevices": self.input_devices,
            "outputDevices": self.output_devices,
            "selectedInputDevice": self.gui_config.sg_input_device,
            "selectedOutputDevice": self.gui_config.sg_output_device,
            "sampleRate": self.gui_config.samplerate or self.get_device_samplerate(safe=True),
            "channels": self.gui_config.channels or self.get_device_channels(safe=True),
        }

    def status_snapshot(self) -> dict:
        return {
            "running": self.running,
            "function": self.gui_config.function,
            "sampleRate": self.gui_config.samplerate,
            "channels": self.gui_config.channels,
            "delayTimeMs": int(self.delay_time_ms),
            "inferTimeMs": int(self.infer_time_ms),
            "selectedHostapi": self.gui_config.sg_hostapi,
            "selectedInputDevice": self.gui_config.sg_input_device,
            "selectedOutputDevice": self.gui_config.sg_output_device,
            "modelName": self.gui_config.model_name,
            "indexPath": self.gui_config.index_path,
            "lastError": self.last_error,
        }

    def update_devices(self, hostapi_name: Optional[str] = None) -> dict:
        with self.lock:
            self.stop_stream()
            try:
                sd._terminate()
            except Exception:
                pass
            try:
                sd._initialize()
            except Exception:
                pass
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            for hostapi in hostapis:
                for device_idx in hostapi["devices"]:
                    devices[device_idx]["hostapi_name"] = hostapi["name"]

            self.hostapis = [hostapi["name"] for hostapi in hostapis]
            if not self.hostapis:
                raise RuntimeError("No audio host APIs available.")

            selected_hostapi = hostapi_name or self.gui_config.sg_hostapi
            if selected_hostapi not in self.hostapis:
                selected_hostapi = self.hostapis[0]

            self.gui_config.sg_hostapi = selected_hostapi
            self.input_devices = [
                d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == selected_hostapi
            ]
            self.output_devices = [
                d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == selected_hostapi
            ]
            self.input_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_input_channels"] > 0 and d["hostapi_name"] == selected_hostapi
            ]
            self.output_devices_indices = [
                d["index"] if "index" in d else d["name"]
                for d in devices
                if d["max_output_channels"] > 0 and d["hostapi_name"] == selected_hostapi
            ]

            if self.input_devices:
                if self.gui_config.sg_input_device not in self.input_devices:
                    self.gui_config.sg_input_device = self.input_devices[0]
            else:
                self.gui_config.sg_input_device = ""

            if self.output_devices:
                if self.gui_config.sg_output_device not in self.output_devices:
                    self.gui_config.sg_output_device = self.output_devices[0]
            else:
                self.gui_config.sg_output_device = ""

            self.gui_config.samplerate = self.get_device_samplerate(safe=True) or 0
            self.gui_config.channels = self.get_device_channels(safe=True) or 1
            return self.devices_snapshot()

    def configure(self, payload: dict, allow_restart: bool = False) -> dict:
        with self.lock:
            restart_required = False

            hostapi = payload.get("hostapi")
            if hostapi and hostapi != self.gui_config.sg_hostapi:
                self.update_devices(hostapi_name=hostapi)
                restart_required = restart_required or self.running

            input_device = payload.get("inputDevice")
            output_device = payload.get("outputDevice")
            if input_device:
                if input_device not in self.input_devices:
                    raise ValueError(f"Input device not found: {input_device}")
                if input_device != self.gui_config.sg_input_device:
                    self.gui_config.sg_input_device = input_device
                    restart_required = restart_required or self.running
            if output_device:
                if output_device not in self.output_devices:
                    raise ValueError(f"Output device not found: {output_device}")
                if output_device != self.gui_config.sg_output_device:
                    self.gui_config.sg_output_device = output_device
                    restart_required = restart_required or self.running

            self.gui_config.wasapi_exclusive = bool(payload.get("wasapiExclusive", self.gui_config.wasapi_exclusive))

            if "function" in payload:
                self.gui_config.function = payload["function"] or "vc"

            if "threshold" in payload:
                self.gui_config.threhold = int(payload["threshold"])
            if "transpose" in payload:
                self.gui_config.pitch = int(round(float(payload["transpose"])))
                if self.rvc is not None:
                    self.rvc.set_key(self.gui_config.pitch)
            if "formant" in payload:
                self.gui_config.formant = float(payload["formant"])
                if self.rvc is not None:
                    self.rvc.set_formant(self.gui_config.formant)
            if "indexRate" in payload:
                self.gui_config.index_rate = float(payload["indexRate"])
                if self.rvc is not None:
                    self.rvc.set_index_rate(self.gui_config.index_rate)
            if "rmsMixRate" in payload:
                self.gui_config.rms_mix_rate = float(payload["rmsMixRate"])
            if "f0Method" in payload:
                self.gui_config.f0method = payload["f0Method"]
            if "inputNoiseReduction" in payload:
                self.gui_config.input_noise_reduce = bool(payload["inputNoiseReduction"])
            if "outputNoiseReduction" in payload:
                self.gui_config.output_noise_reduce = bool(payload["outputNoiseReduction"])
            if "usePhaseVocoder" in payload:
                self.gui_config.use_pv = bool(payload["usePhaseVocoder"])

            for key, attr in [
                ("sampleRateMode", "sr_type"),
                ("sampleLength", "block_time"),
                ("fadeLength", "crossfade_time"),
                ("extraInferenceTime", "extra_time"),
                ("cpuProcesses", "n_cpu"),
            ]:
                if key in payload:
                    new_value = payload[key]
                    if key == "cpuProcesses":
                        new_value = int(new_value)
                    elif key == "sampleRateMode":
                        new_value = str(new_value)
                    else:
                        new_value = float(new_value)
                    if getattr(self.gui_config, attr) != new_value:
                        setattr(self.gui_config, attr, new_value)
                        restart_required = restart_required or self.running

            if "modelName" in payload and payload["modelName"]:
                if self.gui_config.model_name != payload["modelName"]:
                    self.gui_config.model_name = payload["modelName"]
                    restart_required = restart_required or self.running
            if "indexPath" in payload:
                new_index_path = payload["indexPath"] or ""
                if self.gui_config.index_path != new_index_path:
                    self.gui_config.index_path = new_index_path
                    restart_required = restart_required or self.running

            self.set_devices(self.gui_config.sg_input_device, self.gui_config.sg_output_device)

            if restart_required and allow_restart:
                self.stop_stream()
                self._start_realtime_locked()

            return {
                "status": self.status_snapshot(),
                "devices": self.devices_snapshot(),
                "restartRequired": restart_required and not allow_restart,
            }

    def start(self, payload: dict, model_path: str, index_path: str) -> dict:
        with self.lock:
            self.last_error = None
            self.gui_config.model_name = payload["modelName"]
            self.gui_config.index_path = index_path or ""
            self.gui_config.pitch = int(round(float(payload.get("transpose", self.gui_config.pitch))))
            self.gui_config.formant = float(payload.get("formant", self.gui_config.formant))
            self.gui_config.index_rate = float(payload.get("indexRate", self.gui_config.index_rate))
            self.gui_config.rms_mix_rate = float(payload.get("rmsMixRate", self.gui_config.rms_mix_rate))
            self.gui_config.f0method = str(payload.get("f0Method", self.gui_config.f0method))
            self.gui_config.threhold = int(payload.get("threshold", self.gui_config.threhold))
            self.gui_config.block_time = float(payload.get("sampleLength", self.gui_config.block_time))
            self.gui_config.crossfade_time = float(payload.get("fadeLength", self.gui_config.crossfade_time))
            self.gui_config.extra_time = float(payload.get("extraInferenceTime", self.gui_config.extra_time))
            self.gui_config.input_noise_reduce = bool(payload.get("inputNoiseReduction", self.gui_config.input_noise_reduce))
            self.gui_config.output_noise_reduce = bool(payload.get("outputNoiseReduction", self.gui_config.output_noise_reduce))
            self.gui_config.use_pv = bool(payload.get("usePhaseVocoder", self.gui_config.use_pv))
            self.gui_config.n_cpu = int(payload.get("cpuProcesses", self.gui_config.n_cpu))
            self.gui_config.function = str(payload.get("function", self.gui_config.function))
            self.gui_config.sr_type = str(payload.get("sampleRateMode", self.gui_config.sr_type))
            self.gui_config.sg_hostapi = str(payload.get("hostapi", self.gui_config.sg_hostapi or (self.hostapis[0] if self.hostapis else "")))
            self.gui_config.wasapi_exclusive = bool(payload.get("wasapiExclusive", self.gui_config.wasapi_exclusive))
            self.gui_config.sg_input_device = str(payload.get("inputDevice", self.gui_config.sg_input_device))
            self.gui_config.sg_output_device = str(payload.get("outputDevice", self.gui_config.sg_output_device))

            if not model_path:
                raise ValueError("Model path is required.")
            if not Path(model_path).exists():
                raise ValueError("Selected model file does not exist.")

            if not self.gui_config.index_path:
                raise ValueError("Index path is required.")

            if not Path(self.gui_config.index_path).exists():
                raise ValueError("Selected index file does not exist.")

            self.set_devices(self.gui_config.sg_input_device, self.gui_config.sg_output_device)
            self.stop_stream()
            self._start_realtime_locked(model_path=model_path)
            return self.status_snapshot()

    def stop(self) -> dict:
        with self.lock:
            self.stop_stream()
            self.last_error = None
            return self.status_snapshot()

    def _start_realtime_locked(self, model_path: Optional[str] = None) -> None:
        torch.cuda.empty_cache()
        pth_path = model_path or self.gui_config.model_name
        self.rvc = rtrvc.RVC(
            self.gui_config.pitch,
            self.gui_config.formant,
            pth_path,
            self.gui_config.index_path,
            self.gui_config.index_rate,
            self.gui_config.n_cpu,
            self.config.device,
            self.config.use_jit,
            self.config.is_half,
            self.config.dml,
        )
        self.gui_config.samplerate = (
            self.rvc.tgt_sr if self.gui_config.sr_type == "sr_model" else (self.get_device_samplerate() or self.rvc.tgt_sr)
        )
        self.gui_config.channels = self.get_device_channels() or 1
        self.zc = self.gui_config.samplerate // 100
        self.block_frame = int(np.round(self.gui_config.block_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(self.gui_config.crossfade_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(self.gui_config.extra_time * self.gui_config.samplerate / self.zc)) * self.zc
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.gui_config.samplerate, new_freq=16000, dtype=torch.float32
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.gui_config.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr, new_freq=self.gui_config.samplerate, dtype=torch.float32
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9).to(self.config.device)
        self.start_stream()

    def start_stream(self) -> None:
        if self.running:
            return
        extra_settings = None
        if "WASAPI" in self.gui_config.sg_hostapi and self.gui_config.wasapi_exclusive:
            extra_settings = sd.WasapiSettings(exclusive=True)
        self.stream = sd.Stream(
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.gui_config.samplerate,
            channels=self.gui_config.channels,
            dtype="float32",
            extra_settings=extra_settings,
        )
        self.stream.start()
        self.running = True
        self.delay_time_ms = int(
            (
                (self.stream.latency[-1] if self.stream.latency else 0)
                + self.gui_config.block_time
                + self.gui_config.crossfade_time
                + 0.01
                + (min(self.gui_config.crossfade_time, 0.04) if self.gui_config.input_noise_reduce else 0)
            )
            * 1000
        )

    def stop_stream(self) -> None:
        self.running = False
        if self.stream is not None:
            try:
                self.stream.abort()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def audio_callback(self, indata: np.ndarray, outdata: np.ndarray, frames, times, status) -> None:
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        try:
            if self.gui_config.threhold > -60:
                indata = np.append(self.rms_buffer, indata)
                rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
                self.rms_buffer[:] = indata[-4 * self.zc :]
                indata = indata[2 * self.zc - self.zc // 2 :]
                db_threhold = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold
                for i in range(db_threhold.shape[0]):
                    if db_threhold[i]:
                        indata[i * self.zc : (i + 1) * self.zc] = 0
                indata = indata[self.zc // 2 :]

            self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :].clone()
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
            self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()

            if self.gui_config.input_noise_reduce:
                self.input_wav_denoise[:-self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
                input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window
                input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
                self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
                self.nr_buffer[:] = input_wav[self.block_frame :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                    self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                )[160:]
            else:
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = self.resampler(
                    self.input_wav[-indata.shape[0] - 2 * self.zc :]
                )[160:]

            if self.gui_config.function == "vc":
                infer_wav = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    self.gui_config.f0method,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
            elif self.gui_config.input_noise_reduce:
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
            else:
                infer_wav = self.input_wav[self.extra_frame :].clone()

            if self.gui_config.output_noise_reduce and self.gui_config.function == "vc":
                self.output_buffer[:-self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)

            if self.gui_config.rms_mix_rate < 1 and self.gui_config.function == "vc":
                input_wav = self.input_wav_denoise[self.extra_frame :] if self.gui_config.input_noise_reduce else self.input_wav[self.extra_frame :]
                rms1 = librosa.feature.rms(
                    y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms1 = torch.from_numpy(rms1).to(self.config.device)
                rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                rms2 = librosa.feature.rms(
                    y=infer_wav[:].cpu().numpy(),
                    frame_length=4 * self.zc,
                    hop_length=self.zc,
                )
                rms2 = torch.from_numpy(rms2).to(self.config.device)
                rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
                infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate))

            conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(
                F.conv1d(
                    conv_input**2,
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),
                )
                + 1e-8
            )
            if sys.platform == "darwin":
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
                sola_offset = sola_offset.item()
            else:
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])

            infer_wav = infer_wav[sola_offset:]
            if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:
                infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
                infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
            else:
                infer_wav[: self.sola_buffer_frame] = phase_vocoder(
                    self.sola_buffer,
                    infer_wav[: self.sola_buffer_frame],
                    self.fade_out_window,
                    self.fade_in_window,
                )
            self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
            outdata[:] = infer_wav[: self.block_frame].repeat(self.gui_config.channels, 1).t().cpu().numpy()
            self.infer_time_ms = int((time.perf_counter() - start_time) * 1000)
            self.last_error = None
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            self.logger.exception("Realtime VC callback failed")
            outdata[:] = np.zeros_like(outdata)

    def set_devices(self, input_device: str, output_device: str) -> None:
        if input_device:
            sd.default.device[0] = self.input_devices_indices[self.input_devices.index(input_device)]
        if output_device:
            sd.default.device[1] = self.output_devices_indices[self.output_devices.index(output_device)]

    def get_device_samplerate(self, safe: bool = False) -> Optional[int]:
        try:
            return int(sd.query_devices(device=sd.default.device[0])["default_samplerate"])
        except Exception:
            if safe:
                return None
            raise

    def get_device_channels(self, safe: bool = False) -> Optional[int]:
        try:
            max_input_channels = sd.query_devices(device=sd.default.device[0])["max_input_channels"]
            max_output_channels = sd.query_devices(device=sd.default.device[1])["max_output_channels"]
            return min(max_input_channels, max_output_channels, 2)
        except Exception:
            if safe:
                return None
            raise
