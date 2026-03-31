#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import gc
import hashlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import uvicorn

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

default_temp_root = Path(now_dir) / "TEMP"
default_temp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TEMP", str(default_temp_root))
os.environ.setdefault("TMP", str(default_temp_root))
os.environ.setdefault("TMPDIR", str(default_temp_root))

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

from configs import Config
from infer.lib.audio import save_audio
from infer.lib.rvcmd import download_all_assets, sha256
from infer.lib.train.process_ckpt import change_info, extract_small_model, merge
from infer.modules.vc import VC, hash_similarity
from infer.modules.vc.info import show_info, show_model_info
from infer.modules.vc.utils import get_index_path_from_model
from infer.modules.uvr5.modules import uvr, release_uvr_memory
from rvc.onnx import export_onnx as export_rvc_onnx
from realtime_vc import RealtimeVCController
from text_voice_presets import resolve_text_voice_profile


logger = logging.getLogger("phase1_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BACKEND_API_VERSION = "phase1-api-2026-03-29"
BACKEND_BUILD_VERSION = "2026.03.31.2"
BACKEND_SESSION_STARTED_AT = datetime.utcnow().isoformat(timespec="seconds") + "Z"
BACKEND_SESSION_ID = hashlib.sha1(
    f"{BACKEND_BUILD_VERSION}|{BACKEND_SESSION_STARTED_AT}|{os.getpid()}".encode("utf-8")
).hexdigest()[:12]

ROOT = Path(now_dir)
TMP_DIR = ROOT / "TEMP"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "opt"
for directory in (TMP_DIR, LOG_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

config = Config()
vc = VC(config)
engine_lock = threading.RLock()
text_progress_lock = threading.RLock()
selected_model_name: str = ""
realtime_controller: RealtimeVCController | None = None
chattts_runtime = None
chattts_default_speaker = None
chattts_asset_root = ROOT / "asset"


@dataclass
class TextAudioProgressState:
    """持有文本生成任务的运行态与阶段耗时。"""

    active: bool = False
    stage: str = "idle"
    title: str = "Idle"
    detail: str = "Text voice generation is idle."
    completed_steps: int = 0
    total_steps: int = 5
    model_name: str | None = None
    stage_started_monotonic: float | None = None
    run_started_monotonic: float | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)


text_audio_progress_state = TextAudioProgressState()

weight_root = Path(os.getenv("weight_root", "assets/weights"))
index_root = Path(os.getenv("index_root", "logs"))
outside_index_root = Path(os.getenv("outside_index_root", "assets/indices"))
weight_uvr5_root = Path(os.getenv("weight_uvr5_root", "assets/uvr5_weights"))


def _normalize_model_display_name(model_name: str | None) -> str:
    if not model_name:
        return ""
    return Path(model_name).stem or model_name


def _inject_fallback_model_name(summary: str, fallback_model_name: str | None) -> str:
    normalized_name = _normalize_model_display_name(fallback_model_name)
    if not normalized_name:
        return summary

    lines: list[str] = []
    for line in summary.splitlines():
        stripped = line.strip()
        if ":" not in stripped:
            lines.append(line)
            continue

        label, value = stripped.split(":", 1)
        if label.strip() in {"Model name", "模型名"} and value.strip() in {"Unknown", "未知"}:
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}{label}: {normalized_name}")
            continue
        lines.append(line)
    return "\n".join(lines)


def _sanitize_model_info_summary(value: str | None, fallback_model_name: str | None = None) -> tuple[str, str | None]:
    summary = (value or "").strip()
    if not summary:
        return "", None
    if "Traceback (most recent call last):" in summary:
        logger.warning("Suppressing traceback model summary from upstream info helper")
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        detail = next(
            (line for line in reversed(lines) if not line.startswith("Traceback (most recent call last):")),
            "Model metadata inspection failed.",
        )
        return "", detail
    summary = _inject_fallback_model_name(summary, fallback_model_name)
    return summary, None


class SelectModelRequest(BaseModel):
    name: str


class SingleInferencePayload(BaseModel):
    modelName: str
    inputFileURL: str
    outputDirectoryURL: str
    speakerId: int = 0
    transpose: float
    f0Method: str
    indexPath: Optional[str] = None
    indexRate: float
    filterRadius: float
    resampleSR: float
    rmsMixRate: float
    protect: float
    f0FileURL: Optional[str] = None


class TextAudioPayload(BaseModel):
    modelName: str
    text: str
    outputDirectoryURL: str
    gender: str = "female"
    toneMode: str = "preset"
    tonePreset: Optional[str] = None
    customToneText: str = ""
    matchProfile: str = "identityLock"
    speakerId: int = 0
    transpose: float
    speechRate: Optional[str] = None
    f0Method: str
    indexPath: Optional[str] = None
    indexRate: float
    filterRadius: float
    resampleSR: float
    rmsMixRate: float
    protect: float


class BatchInferencePayload(BaseModel):
    modelName: str
    inputDirectoryURL: Optional[str] = None
    inputFileURLs: list[str] = []
    outputDirectoryURL: str
    format: str
    speakerId: int = 0
    transpose: float
    f0Method: str
    indexPath: Optional[str] = None
    indexRate: float
    filterRadius: float
    resampleSR: float
    rmsMixRate: float
    protect: float


class UVRConvertPayload(BaseModel):
    modelName: str
    inputDirectoryURL: Optional[str] = None
    inputFileURLs: list[str] = []
    vocalOutputDirectoryURL: str
    instrumentalOutputDirectoryURL: str
    format: str = "wav"
    agg: int = 10


class ExportOnnxPayload(BaseModel):
    modelPath: str
    onnxOutputPath: str


class CheckpointComparePayload(BaseModel):
    modelIDA: str
    modelIDB: str


class CheckpointInfoPayload(BaseModel):
    modelPath: str


class CheckpointModifyPayload(BaseModel):
    modelPath: str
    infoText: str
    saveName: str = ""


class CheckpointMergePayload(BaseModel):
    modelPathA: str
    modelPathB: str
    weightA: float = 0.5
    targetSampleRate: str = "48k"
    hasPitchGuidance: bool = True
    infoText: str = ""
    saveName: str
    version: str = "v2"


class CheckpointExtractPayload(BaseModel):
    modelPath: str
    saveName: str
    author: str = ""
    targetSampleRate: str = "48k"
    hasPitchGuidance: bool = True
    infoText: str = ""
    version: str = "v2"


class RealtimeConfigurePayload(BaseModel):
    hostapi: Optional[str] = None
    inputDevice: Optional[str] = None
    outputDevice: Optional[str] = None
    wasapiExclusive: Optional[bool] = None
    function: Optional[str] = None
    threshold: Optional[int] = None
    transpose: Optional[float] = None
    formant: Optional[float] = None
    indexRate: Optional[float] = None
    rmsMixRate: Optional[float] = None
    f0Method: Optional[str] = None
    inputNoiseReduction: Optional[bool] = None
    outputNoiseReduction: Optional[bool] = None
    usePhaseVocoder: Optional[bool] = None
    sampleRateMode: Optional[str] = None
    sampleLength: Optional[float] = None
    fadeLength: Optional[float] = None
    extraInferenceTime: Optional[float] = None
    cpuProcesses: Optional[int] = None
    modelName: Optional[str] = None
    indexPath: Optional[str] = None


class RealtimeStartPayload(RealtimeConfigurePayload):
    modelName: str


def local_path(value: str | None) -> str | None:
    if not value:
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        return unquote(parsed.path)
    return value


def get_realtime_controller() -> RealtimeVCController:
    global realtime_controller
    if realtime_controller is None:
        realtime_controller = RealtimeVCController(config, logger)
    return realtime_controller


def get_chattts_runtime():
    global chattts_runtime, chattts_default_speaker

    try:
        import ChatTTS  # type: ignore
        import torch
        import torchaudio
    except Exception as exc:  # pragma: no cover - runtime dependency gate
        raise HTTPException(
            status_code=500,
            detail="ChatTTS is not installed in the engine environment. Re-run the engine dependency install to enable text voice generation.",
        ) from exc

    if chattts_runtime is None:
        logger.info("Loading ChatTTS runtime")
        chat = ChatTTS.Chat()
        try:
            loaded = chat.load(
                source="custom",
                custom_path=str(chattts_asset_root),
                compile=False,
            )
        except TypeError:
            loaded = chat.load(
                source="custom",
                custom_path=str(chattts_asset_root),
            )
        if not loaded:
            raise HTTPException(
                status_code=500,
                detail="ChatTTS assets are missing or incomplete. Refresh the local ChatTTS asset bundle before running text voice generation.",
            )
        chattts_runtime = {
            "module": ChatTTS,
            "chat": chat,
            "torch": torch,
            "torchaudio": torchaudio,
        }

    if chattts_default_speaker is None:
        chat = chattts_runtime["chat"]
        if hasattr(chat, "speaker") and getattr(chat, "speaker", None) is not None and hasattr(chat, "sample_random_speaker"):
            chattts_default_speaker = chat.sample_random_speaker()

    return chattts_runtime


def _clear_torch_runtime_caches() -> None:
    """在文本任务边界主动回收 Python 与 torch 缓存，减少重复生成后的内存爬升。"""

    gc.collect()
    try:
        import torch

        if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def release_chattts_runtime(detail: str | None = None) -> None:
    """完整释放 ChatTTS runtime，避免长驻状态污染后续文本任务。"""

    global chattts_runtime, chattts_default_speaker

    runtime = chattts_runtime
    chattts_runtime = None
    chattts_default_speaker = None

    if runtime is not None:
        chat = runtime.get("chat")
        if chat is not None and hasattr(chat, "unload"):
            try:
                chat.unload()
            except Exception:
                logger.warning("ChatTTS unload failed during runtime cleanup", exc_info=True)

    _clear_torch_runtime_caches()

    if detail:
        logger.info(detail)


def reset_task_runtime(task_kind: str, release_chattts: bool = False) -> None:
    """在每轮任务结束后清理任务级缓存，但保留当前选中的 RVC 模型。"""

    if release_chattts:
        release_chattts_runtime(f"Released ChatTTS runtime after {task_kind}.")

    with engine_lock:
        if getattr(vc, "hubert_model", None) is not None:
            vc.hubert_model = None
        if getattr(vc, "pipeline", None) is not None:
            vc.pipeline = None

    _clear_torch_runtime_caches()
    logger.info("Reset transient runtime after %s.", task_kind)


def _stable_text_audio_seed(text: str, payload: TextAudioPayload, profile_id: str) -> int:
    """对同一文本请求生成稳定 seed，减少重复点击时的随机漂移。"""

    fingerprint = "|".join(
        [
            payload.modelName.strip(),
            text.strip(),
            payload.gender.strip(),
            payload.toneMode.strip(),
            payload.tonePreset or "",
            payload.customToneText.strip(),
            payload.matchProfile.strip(),
            payload.speechRate.strip(),
            payload.f0Method.strip(),
            f"{payload.transpose:.4f}",
            f"{payload.indexRate:.4f}",
            f"{payload.filterRadius:.4f}",
            f"{payload.rmsMixRate:.4f}",
            f"{payload.protect:.4f}",
            profile_id.strip(),
        ]
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def set_text_audio_progress(
    stage: str,
    title: str,
    detail: str,
    completed_steps: int,
    total_steps: int = 5,
    active: bool = True,
    model_name: str | None = None,
) -> None:
    now = time.monotonic()
    with text_progress_lock:
        state = text_audio_progress_state
        if state.run_started_monotonic is None or (stage == "preparing" and completed_steps == 0):
            state.run_started_monotonic = now
            state.stage_started_monotonic = now if active else None
            state.stage_durations = {}
        elif state.stage != stage and state.stage_started_monotonic is not None:
            previous_stage = state.stage
            if previous_stage not in {"idle", "completed", "failed"}:
                state.stage_durations[previous_stage] = max(now - state.stage_started_monotonic, 0.0)
            state.stage_started_monotonic = now if active else None
        elif state.stage_started_monotonic is None and active:
            state.stage_started_monotonic = now

        state.active = active
        state.stage = stage
        state.title = title
        state.detail = detail
        state.completed_steps = completed_steps
        state.total_steps = total_steps
        state.model_name = model_name
        if not active:
            state.stage_started_monotonic = None


def reset_text_audio_progress(detail: str = "Text voice generation is idle.") -> None:
    with text_progress_lock:
        text_audio_progress_state.active = False
        text_audio_progress_state.stage = "idle"
        text_audio_progress_state.title = "Idle"
        text_audio_progress_state.detail = detail
        text_audio_progress_state.completed_steps = 0
        text_audio_progress_state.total_steps = 5
        text_audio_progress_state.model_name = None
        text_audio_progress_state.stage_started_monotonic = None
        text_audio_progress_state.run_started_monotonic = None
        text_audio_progress_state.stage_durations = {}


def get_text_audio_progress_snapshot() -> dict:
    now = time.monotonic()
    with text_progress_lock:
        state = text_audio_progress_state
        stage_elapsed = 0.0
        total_elapsed = 0.0
        if state.stage_started_monotonic is not None:
            stage_elapsed = max(now - state.stage_started_monotonic, 0.0)
        if state.run_started_monotonic is not None:
            total_elapsed = max(now - state.run_started_monotonic, 0.0)
        return {
            "active": state.active,
            "stage": state.stage,
            "title": state.title,
            "detail": state.detail,
            "completedSteps": state.completed_steps,
            "totalSteps": state.total_steps,
            "modelName": state.model_name,
            "stageElapsedSeconds": stage_elapsed,
            "totalElapsedSeconds": total_elapsed,
            "stageDurations": dict(state.stage_durations),
        }


def _segment_text_for_tts(text: str, payload: TextAudioPayload) -> list[str]:
    """仅按大句切分文本，把逗号留在句内由模型自己处理，减少边界断气。"""

    del payload

    strong_breaks = set("。！？!?；;")

    segments: list[str] = []
    buffer: list[str] = []

    for char in text:
        buffer.append(char)
        if char in strong_breaks:
            segment = "".join(buffer).strip()
            if segment:
                segments.append(segment)
            buffer = []

    trailing = "".join(buffer).strip()
    if trailing:
        segments.append(trailing)

    if not segments:
        return [text]
    return segments


def _pause_seconds_for_segment(segment: str, payload: TextAudioPayload) -> float:
    """根据标点和当前风格返回拼接段落之间的停顿时长。"""

    last_char = segment.rstrip()[-1] if segment.strip() else ""
    normalized_preset = (payload.tonePreset or "").strip().lower()
    normalized_rate = payload.speechRate.strip().lower()

    comma_like = 0.14
    sentence_like = 0.48
    colon_like = 0.24

    if normalized_rate == "slow":
        comma_like = 0.20
        sentence_like = 0.62
        colon_like = 0.32

    if normalized_preset == "female_sad_youth" and normalized_rate == "slow":
        comma_like = 0.22
        sentence_like = 0.84
        colon_like = 0.46
    elif normalized_preset == "female_heartbroken" and normalized_rate == "slow":
        comma_like = 0.24
        sentence_like = 0.90
        colon_like = 0.46
    elif normalized_preset == "female_gentle" and normalized_rate == "slow":
        comma_like = 0.18
        sentence_like = 0.74
        colon_like = 0.38
    elif normalized_preset == "female_sad_youth":
        sentence_like = max(sentence_like, 0.64)
    elif normalized_preset == "female_heartbroken":
        sentence_like = max(sentence_like, 0.72)
    elif normalized_preset == "female_gentle":
        sentence_like = max(sentence_like, 0.58)

    if last_char in "。！？!?":
        return sentence_like
    if last_char in "：:":
        return colon_like
    if last_char in "，,；;":
        return comma_like
    return 0.0


def _uses_breath_transition(segment: str, payload: TextAudioPayload) -> bool:
    """只在需要文学感停顿的女性预设上，为句号级停顿加入极轻的换气过渡。"""

    if not segment.strip():
        return False

    if segment.rstrip()[-1] not in "。！？!?":
        return False

    normalized_preset = (payload.tonePreset or "").strip().lower()
    normalized_rate = payload.speechRate.strip().lower()
    if normalized_preset not in {"female_sad_youth", "female_heartbroken", "female_gentle"}:
        return False
    return normalized_rate in {"medium", "slow"}


def _build_pause_audio(segment: str, payload: TextAudioPayload, sample_rate: int = 24000) -> np.ndarray:
    """构造句间停顿音频，必要时在前沿加入轻微换气感而不是纯静音断层。"""

    pause_seconds = _pause_seconds_for_segment(segment, payload)
    if pause_seconds <= 0:
        return np.zeros(0, dtype=np.float32)

    total_samples = int(sample_rate * pause_seconds)
    if total_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    pause_audio = np.zeros(total_samples, dtype=np.float32)
    if not _uses_breath_transition(segment, payload):
        return pause_audio

    boundary_silence_seconds = min(0.05, pause_seconds * 0.12)
    prefix_silence_samples = int(sample_rate * boundary_silence_seconds)
    suffix_silence_samples = int(sample_rate * boundary_silence_seconds)

    available_breath_samples = total_samples - prefix_silence_samples - suffix_silence_samples
    if available_breath_samples <= 8:
        return pause_audio

    breath_seconds = min(0.16, max(0.08, pause_seconds * 0.22))
    breath_samples = min(int(sample_rate * breath_seconds), available_breath_samples)
    if breath_samples <= 8:
        return pause_audio

    seed_material = f"{segment}|{payload.tonePreset}|{payload.speechRate}"
    breath_seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(breath_seed)
    breath_noise = rng.normal(0.0, 1.0, breath_samples).astype(np.float32)

    # 用高通感噪声和缓入缓出的包络模拟轻换气，只求过渡自然，不追求真实气口拟真。
    breath_noise[1:] = breath_noise[1:] - 0.82 * breath_noise[:-1]
    envelope = np.power(
        np.clip(np.sin(np.linspace(0.0, np.pi, breath_samples, dtype=np.float32)), 0.0, 1.0),
        1.6,
    )
    breath_audio = breath_noise * envelope * 0.0025
    start_index = prefix_silence_samples
    end_index = start_index + breath_samples
    pause_audio[start_index:end_index] = breath_audio
    return pause_audio


def synthesize_text_to_temp_wav(text: str, payload: TextAudioPayload) -> Path:
    trimmed_text = text.strip()
    if not trimmed_text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    set_text_audio_progress(
        stage="loadingChatTTS",
        title="Load ChatTTS",
        detail="Loading ChatTTS runtime and voice seed.",
        completed_steps=1,
    )
    runtime = get_chattts_runtime()
    ChatTTS = runtime["module"]
    chat = runtime["chat"]
    torch = runtime["torch"]
    torchaudio = runtime["torchaudio"]
    profile = resolve_text_voice_profile(
        payload.gender,
        payload.toneMode,
        payload.tonePreset,
        payload.customToneText,
        payload.speechRate,
    )
    manual_seed = _stable_text_audio_seed(trimmed_text, payload, profile.resolved_preset_id)
    segments = _segment_text_for_tts(trimmed_text, payload)

    infer_kwargs = {}
    if hasattr(ChatTTS.Chat, "InferCodeParams"):
        infer_kwargs["params_infer_code"] = ChatTTS.Chat.InferCodeParams(
            spk_emb=profile.spk_emb or chattts_default_speaker,
            prompt=profile.prompt,
            temperature=profile.temperature,
            top_P=profile.top_p,
            top_K=profile.top_k,
            manual_seed=manual_seed,
        )
    if hasattr(ChatTTS.Chat, "RefineTextParams") and profile.refine_prompt.strip():
        infer_kwargs["params_refine_text"] = ChatTTS.Chat.RefineTextParams(
            prompt=profile.refine_prompt,
            manual_seed=manual_seed,
        )

    set_text_audio_progress(
        stage="generatingSpeech",
        title="Generate speech",
        detail="ChatTTS is synthesizing a literal readout source speech track.",
        completed_steps=2,
    )
    try:
        wavs = chat.infer(
            segments,
            skip_refine_text=True,
            do_text_normalization=False,
            do_homophone_replacement=False,
            split_text=False,
            **infer_kwargs,
        )
    except Exception as exc:
        logger.exception("ChatTTS generation failed")
        raise HTTPException(status_code=500, detail=f"ChatTTS failed: {exc}") from exc

    if not wavs:
        raise HTTPException(status_code=500, detail="ChatTTS returned no audio.")

    stitched_segments: list[np.ndarray] = []
    for index, wav in enumerate(wavs):
        if hasattr(wav, "detach"):
            wav_array = wav.detach().float().cpu().numpy().reshape(-1)
        else:
            wav_array = np.asarray(wav, dtype=np.float32).reshape(-1)
        stitched_segments.append(wav_array.astype(np.float32, copy=False))

        if index < len(segments) - 1:
            pause_audio = _build_pause_audio(segments[index], payload)
            if pause_audio.size > 0:
                stitched_segments.append(pause_audio)

    stitched_wav = np.concatenate(stitched_segments).astype(np.float32, copy=False)
    wav_tensor = torch.from_numpy(stitched_wav.reshape(1, -1))

    temp_dir = Path(tempfile.mkdtemp(prefix="text-audio-", dir=str(TMP_DIR)))
    source_path = temp_dir / "tts-source.wav"
    torchaudio.save(str(source_path), wav_tensor, 24000)
    return source_path


def scan_models() -> list[str]:
    if not weight_root.exists():
        return []
    return sorted(
        item.name
        for item in weight_root.iterdir()
        if item.is_file() and item.suffix == ".pth"
    )


def scan_indices() -> list[str]:
    paths: list[str] = []
    for base in (index_root, outside_index_root):
        if not base.exists():
            continue
        for item in base.rglob("*.index"):
            if "trained" not in item.name:
                paths.append(str(item))
    return sorted(set(paths))


def scan_uvr_models() -> list[str]:
    if not weight_uvr5_root.exists():
        return []

    model_names: list[str] = []
    for item in weight_uvr5_root.iterdir():
        if item.is_file() and item.suffix == ".pth":
            model_names.append(item.name.replace(".pth", ""))
        elif item.is_dir() and "onnx" in item.name:
            model_names.append(item.name)
    return sorted(set(model_names))


def asset_integrity_specs() -> list[dict]:
    specs = [
        {
            "title": "Hubert base",
            "path": "assets/hubert/hubert_base.pt",
            "env": "sha256_hubert_base_pt",
            "note": "Required for inference and training.",
        },
        {
            "title": "RMVPE weights",
            "path": "assets/rmvpe/rmvpe.pt",
            "env": "sha256_rmvpe_pt",
            "note": "Required for rmvpe pitch extraction.",
        },
        {
            "title": "RMVPE ONNX",
            "path": "assets/rmvpe/rmvpe.onnx",
            "env": "sha256_rmvpe_onnx",
            "note": "Required for DML and non-default rmvpe paths.",
        },
    ]

    model_names = [
        "D32k.pth",
        "D40k.pth",
        "D48k.pth",
        "G32k.pth",
        "G40k.pth",
        "G48k.pth",
        "f0D32k.pth",
        "f0D40k.pth",
        "f0D48k.pth",
        "f0G32k.pth",
        "f0G40k.pth",
        "f0G48k.pth",
    ]
    for model_name in model_names:
        env_name = model_name.replace(".", "_")
        specs.append(
            {
                "title": f"Pretrained v1 {model_name}",
                "path": f"assets/pretrained/{model_name}",
                "env": f"sha256_v1_{env_name}",
                "note": "Required for several v1 training flows.",
            }
        )
        specs.append(
            {
                "title": f"Pretrained v2 {model_name}",
                "path": f"assets/pretrained_v2/{model_name}",
                "env": f"sha256_v2_{env_name}",
                "note": "Required for several v2 training flows.",
            }
        )

    uvr_models = [
        "HP2-人声vocals+非人声instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-主旋律人声vocals+其他instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth",
    ]
    for model_name in uvr_models:
        env_name = model_name.replace(".", "_")
        specs.append(
            {
                "title": f"UVR5 {model_name}",
                "path": f"assets/uvr5_weights/{model_name}",
                "env": f"sha256_uvr5_{env_name}",
                "note": "Required for upstream UVR workflows.",
            }
        )

    specs.extend(
        [
            {
                "title": "UVR5 dereverb ONNX",
                "path": "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
                "env": "sha256_uvr5_vocals_onnx",
                "note": "Required for dereverb-capable UVR models.",
            },
            {
                "title": "Weights folder",
                "path": "assets/weights",
                "env": None,
                "note": "Current inference models location.",
            },
            {
                "title": "Indices folder",
                "path": "assets/indices",
                "env": None,
                "note": "Current index override location.",
            },
        ]
    )
    return specs


def asset_integrity_report() -> dict:
    items: list[dict] = []
    all_valid = True

    for spec in asset_integrity_specs():
        target = ROOT / spec["path"]
        expected_hash = os.environ.get(spec["env"]) if spec["env"] else None
        actual_hash = None

        if not target.exists():
            status = "missing"
            all_valid = False
        elif target.is_dir():
            status = "ok"
        elif expected_hash is None:
            status = "error"
            all_valid = False
        else:
            with open(target, "rb") as handle:
                actual_hash = sha256(handle)
            if actual_hash == expected_hash:
                status = "ok"
            else:
                status = "mismatch"
                all_valid = False

        items.append(
            {
                "title": spec["title"],
                "path": spec["path"],
                "status": status,
                "note": spec["note"],
                "expectedHash": expected_hash,
                "actualHash": actual_hash,
            }
        )

    verified_count = sum(1 for item in items if item["status"] == "ok")
    return {
        "items": items,
        "allValid": all_valid,
        "checkedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "message": f"{verified_count}/{len(items)} tracked assets verified",
    }


def download_assets() -> dict:
    temp_dir = Path(tempfile.mkdtemp(prefix="asset-update-", dir=str(TMP_DIR)))
    try:
        with engine_lock:
            download_all_assets(str(temp_dir))
        report = asset_integrity_report()
    except Exception as exc:
        logger.exception("Asset download failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    message = "Asset download completed and all tracked assets verified."
    if not report["allValid"]:
        message = "Asset download completed, but some tracked assets still failed verification."
    return {
        "message": message,
        "report": report,
    }


def ensure_model_loaded(model_name: str) -> dict:
    global selected_model_name
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required.")
    try:
        with engine_lock:
            if selected_model_name != model_name or vc.net_g is None:
                model_state = vc.get_vc(model_name, 0.33, 0.33, "", "")
                selected_model_name = model_name
            else:
                model_state = (
                    {"visible": True, "maximum": vc.n_spk or 0, "__type__": "update"},
                    {"visible": vc.if_f0 != 0, "value": 0.33, "__type__": "update"},
                    {"visible": vc.if_f0 != 0, "value": 0.33, "__type__": "update"},
                    {"value": get_index_path_from_model(model_name), "__type__": "update"},
                    {"value": get_index_path_from_model(model_name), "__type__": "update"},
                    show_model_info(vc.cpt),
                )
    except ValueError as exc:
        logger.warning("Model load rejected for %s: %s", model_name, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model load failed for %s", model_name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    model_info_summary, model_info_error = _sanitize_model_info_summary(
        model_state[5] if len(model_state) > 5 else "",
        fallback_model_name=model_name,
    )
    return {
        "modelName": model_name,
        "modelInfoSummary": model_info_summary,
        "modelInfoError": model_info_error,
        "indexPaths": [item for item in scan_indices() if item],
        "speakerCount": int(vc.n_spk or 0),
    }


def unload_model() -> dict:
    # Clear the active VC model and any realtime session state so the client can refresh from a clean slate.
    global selected_model_name, realtime_controller
    with engine_lock:
        if realtime_controller is not None:
            if realtime_controller.running:
                realtime_controller.stop()
            realtime_controller.gui_config.model_name = ""
            realtime_controller.gui_config.index_path = ""
            realtime_controller.last_error = None

        # Reuse the upstream empty-sid cache-clearing path.
        vc.get_vc("", 0.33, 0.33, "", "")
        vc.pipeline = None
        vc.cpt = None
        vc.if_f0 = None
        vc.version = None
        vc.net_g = None
        vc.n_spk = None
        vc.tgt_sr = None
        vc.hubert_model = None
        selected_model_name = ""

    return {
        "modelName": "",
        "modelInfoSummary": "",
        "indexPaths": [item for item in scan_indices() if item],
        "speakerCount": 0,
        "unloaded": True,
    }


def _normalize_local_path(value: str | None) -> Path:
    path_value = local_path(value)
    if not path_value:
        raise HTTPException(status_code=400, detail="Path is required.")
    return Path(path_value)


def _stage_uvr_inputs(payload: UVRConvertPayload) -> tuple[Path, list[str]]:
    if bool(payload.inputDirectoryURL) == bool(payload.inputFileURLs):
        raise HTTPException(
            status_code=400,
            detail="Provide either inputDirectoryURL or inputFileURLs, but not both.",
        )

    source_paths: list[Path] = []
    if payload.inputDirectoryURL:
        input_directory = _normalize_local_path(payload.inputDirectoryURL)
        if not input_directory.exists() or not input_directory.is_dir():
            raise HTTPException(status_code=400, detail="Input directory does not exist.")
        source_paths = [item for item in sorted(input_directory.iterdir()) if item.is_file()]
    else:
        source_paths = []
        for item in payload.inputFileURLs:
            file_path = _normalize_local_path(item)
            if not file_path.exists() or not file_path.is_file():
                raise HTTPException(status_code=400, detail="One or more input files do not exist.")
            source_paths.append(file_path)

    if not source_paths:
        raise HTTPException(status_code=400, detail="No UVR input files were provided.")

    temp_dir = Path(tempfile.mkdtemp(prefix="uvr-", dir=str(TMP_DIR)))
    stage_dir = temp_dir / "input"
    stage_dir.mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    for index, source in enumerate(source_paths):
        stage_name = source.name
        if stage_name in seen_names:
            stage_name = f"{index:03d}_{stage_name}"
        seen_names.add(stage_name)
        shutil.copy2(source, stage_dir / stage_name)

    return stage_dir, [item.name for item in stage_dir.iterdir() if item.is_file()]


def run_uvr(payload: UVRConvertPayload) -> dict:
    model_names = scan_uvr_models()
    if payload.modelName not in model_names:
        raise HTTPException(status_code=400, detail="Selected UVR model is not available.")

    format_name = payload.format.lower().strip()
    if format_name not in {"wav", "flac", "mp3", "m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported UVR export format.")

    vocal_output_root = _normalize_local_path(payload.vocalOutputDirectoryURL)
    instrumental_output_root = _normalize_local_path(payload.instrumentalOutputDirectoryURL)
    vocal_output_root.mkdir(parents=True, exist_ok=True)
    instrumental_output_root.mkdir(parents=True, exist_ok=True)

    stage_dir, staged_paths = _stage_uvr_inputs(payload)
    message = ""

    try:
        with engine_lock:
            for item in uvr(
                payload.modelName,
                str(stage_dir),
                str(vocal_output_root),
                [],
                str(instrumental_output_root),
                payload.agg,
                format_name,
            ):
                message = item
    except Exception as exc:
        logger.exception("UVR conversion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        shutil.rmtree(stage_dir.parent, ignore_errors=True)

    return {
        "message": message,
        "vocalOutputDirectoryURL": vocal_output_root.as_uri(),
        "instrumentalOutputDirectoryURL": instrumental_output_root.as_uri(),
        "vocalOutputFileURLs": [
            path.as_uri()
            for path in sorted(vocal_output_root.iterdir())
            if path.is_file()
        ],
        "instrumentalOutputFileURLs": [
            path.as_uri()
            for path in sorted(instrumental_output_root.iterdir())
            if path.is_file()
        ],
        "modelName": payload.modelName,
        "stagedInputCount": len(staged_paths),
    }


def release_uvr_runtime() -> dict:
    try:
        with engine_lock:
            release_uvr_memory()
    except Exception as exc:
        logger.exception("UVR memory release failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"released": True, "message": "UVR memory released."}


def release_runtime_memory() -> dict:
    global selected_model_name, realtime_controller, chattts_runtime, chattts_default_speaker
    try:
        with engine_lock:
            realtime_released = False
            if realtime_controller is not None:
                realtime_controller.release_runtime()
                realtime_released = True

            model_result = unload_model()
            release_uvr_memory()
            release_chattts_runtime()
            reset_text_audio_progress("Text voice generation cache was released.")

            selected_model_name = ""

        return {
            "released": True,
            "message": "Runtime memory released.",
            "modelUnloaded": model_result.get("unloaded", False),
            "realtimeReleased": realtime_released,
            "uvrReleased": True,
        }
    except Exception as exc:
        logger.exception("Runtime memory release failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_export_onnx(payload: ExportOnnxPayload) -> dict:
    model_path = _normalize_local_path(payload.modelPath)
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=400, detail="Model file does not exist.")

    onnx_output_path = _normalize_local_path(payload.onnxOutputPath)
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with engine_lock:
            message = export_rvc_onnx(str(model_path), str(onnx_output_path))
    except Exception as exc:
        logger.exception("ONNX export failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": message or "ONNX export completed.",
        "modelPath": model_path.as_uri(),
        "exportedPath": onnx_output_path.as_uri(),
    }


def run_checkpoint_compare(payload: CheckpointComparePayload) -> dict:
    model_id_a = payload.modelIDA.strip()
    model_id_b = payload.modelIDB.strip()
    if not model_id_a or not model_id_b:
        raise HTTPException(status_code=400, detail="Both model IDs are required.")

    try:
        with engine_lock:
            similarity = hash_similarity(model_id_a, model_id_b)
    except Exception as exc:
        logger.exception("Checkpoint comparison failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if isinstance(similarity, str):
        similarity_text = similarity
    else:
        similarity_text = f"{float(similarity):.4f}"

    return {
        "message": f"Model similarity: {similarity_text}",
        "similarity": similarity_text,
    }


def run_checkpoint_show(payload: CheckpointInfoPayload) -> dict:
    model_path = _normalize_local_path(payload.modelPath)
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=400, detail="Model file does not exist.")

    try:
        with engine_lock:
            info_text = show_info(str(model_path))
    except Exception as exc:
        logger.exception("Checkpoint info read failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "Checkpoint metadata loaded.",
        "infoText": info_text,
        "modelPath": model_path.as_uri(),
    }


def run_checkpoint_modify(payload: CheckpointModifyPayload) -> dict:
    model_path = _normalize_local_path(payload.modelPath)
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=400, detail="Model file does not exist.")

    save_name = payload.saveName.strip()
    if not save_name:
        save_name = model_path.name

    try:
        with engine_lock:
            message = change_info(str(model_path), payload.infoText, save_name)
    except Exception as exc:
        logger.exception("Checkpoint info modify failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": message,
        "outputModelPath": (weight_root / save_name).as_uri(),
    }


def run_checkpoint_merge(payload: CheckpointMergePayload) -> dict:
    model_path_a = _normalize_local_path(payload.modelPathA)
    model_path_b = _normalize_local_path(payload.modelPathB)
    if not model_path_a.exists() or not model_path_a.is_file():
        raise HTTPException(status_code=400, detail="Model A file does not exist.")
    if not model_path_b.exists() or not model_path_b.is_file():
        raise HTTPException(status_code=400, detail="Model B file does not exist.")

    save_name = payload.saveName.strip()
    if not save_name:
        raise HTTPException(status_code=400, detail="Save name is required.")

    try:
        with engine_lock:
            message = merge(
                str(model_path_a),
                str(model_path_b),
                float(payload.weightA),
                payload.targetSampleRate,
                "Yes" if payload.hasPitchGuidance else "No",
                payload.infoText,
                save_name,
                payload.version,
            )
    except Exception as exc:
        logger.exception("Checkpoint merge failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": message,
        "outputModelPath": (weight_root / f"{save_name}.pth").as_uri(),
    }


def run_checkpoint_extract(payload: CheckpointExtractPayload) -> dict:
    model_path = _normalize_local_path(payload.modelPath)
    if not model_path.exists() or not model_path.is_file():
        raise HTTPException(status_code=400, detail="Training checkpoint file does not exist.")

    save_name = payload.saveName.strip()
    if not save_name:
        raise HTTPException(status_code=400, detail="Save name is required.")

    try:
        with engine_lock:
            message = extract_small_model(
                str(model_path),
                save_name,
                payload.author,
                payload.targetSampleRate,
                "1" if payload.hasPitchGuidance else "0",
                payload.infoText,
                payload.version,
            )
    except Exception as exc:
        logger.exception("Checkpoint extract failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": message,
        "outputModelPath": (weight_root / f"{save_name}.pth").as_uri(),
    }


def single_output_path(input_path: str) -> Path:
    stem = Path(input_path).stem or "converted"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{stem}-{stamp}.wav"


def single_output_path_in_directory(input_path: str, output_root: Path) -> Path:
    stem = Path(input_path).stem or "converted"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / f"{stem}-{stamp}.wav"


def text_output_path_in_directory(text: str, output_root: Path) -> Path:
    slug = "".join(char if char.isalnum() else "-" for char in text.strip()[:24]).strip("-")
    slug = "-".join(filter(None, slug.split("-"))).lower() or "text-audio"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / f"{slug}-{stamp}.wav"


def text_source_path_in_directory(text: str, output_root: Path) -> Path:
    slug = "".join(char if char.isalnum() else "-" for char in text.strip()[:24]).strip("-")
    slug = "-".join(filter(None, slug.split("-"))).lower() or "text-audio"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_root / f"{slug}-{stamp}-source.wav"


def run_single(payload: SingleInferencePayload) -> dict:
    output: tuple[int, np.ndarray] | None = None
    try:
        ensure_model_loaded(payload.modelName)
        input_path = local_path(payload.inputFileURL)
        if not input_path or not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail="Input audio file does not exist.")
        output_root = Path(local_path(payload.outputDirectoryURL) or payload.outputDirectoryURL)
        output_root.mkdir(parents=True, exist_ok=True)

        index_path = payload.indexPath or get_index_path_from_model(payload.modelName)
        f0_file = local_path(payload.f0FileURL)

        with engine_lock:
            message, output = vc.vc_single(
                payload.speakerId,
                input_path,
                payload.transpose,
                f0_file,
                payload.f0Method,
                "",
                index_path or "",
                payload.indexRate,
                payload.filterRadius,
                payload.resampleSR,
                payload.rmsMixRate,
                payload.protect,
            )

        if output is None:
            raise HTTPException(status_code=400, detail=message)

        target_sr, audio_opt = output
        output_path = single_output_path_in_directory(input_path, output_root)
        save_audio(str(output_path), audio_opt, target_sr, f32=False, format="wav")
        return {
            "message": message,
            "outputAudioURL": output_path.as_uri(),
            "outputDirectoryURL": output_root.as_uri(),
        }
    finally:
        output = None
        reset_task_runtime("single conversion")


def run_text(payload: TextAudioPayload) -> dict:
    source_path: Path | None = None
    preserved_source_path: Path | None = None
    output: tuple[int, np.ndarray] | None = None
    set_text_audio_progress(
        stage="preparing",
        title="Prepare task",
        detail="Validating text input and reserving the output path.",
        completed_steps=0,
        model_name=payload.modelName,
    )
    ensure_model_loaded(payload.modelName)
    trimmed_text = payload.text.strip()
    if not trimmed_text:
        reset_text_audio_progress("Text voice generation failed.")
        raise HTTPException(status_code=400, detail="Text input is required.")
    if payload.toneMode.strip().lower() == "custom" and not payload.customToneText.strip():
        reset_text_audio_progress("Text voice generation failed.")
        raise HTTPException(status_code=400, detail="Custom tone mode requires a custom tone description.")

    output_root = Path(local_path(payload.outputDirectoryURL) or payload.outputDirectoryURL)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = payload.indexPath or get_index_path_from_model(payload.modelName)

    try:
        source_path = synthesize_text_to_temp_wav(trimmed_text, payload)
        preserved_source_path = text_source_path_in_directory(trimmed_text, output_root)
        shutil.copy2(source_path, preserved_source_path)
        set_text_audio_progress(
            stage="convertingVoice",
            title="Convert voice",
            detail="Applying the selected RVC voice model to the generated speech.",
            completed_steps=3,
            model_name=payload.modelName,
        )
        with engine_lock:
            message, output = vc.vc_single(
                payload.speakerId,
                str(source_path),
                payload.transpose,
                None,
                payload.f0Method,
                "",
                index_path or "",
                payload.indexRate,
                payload.filterRadius,
                payload.resampleSR,
                payload.rmsMixRate,
                payload.protect,
            )
        if output is None:
            set_text_audio_progress(
                stage="failed",
                title="Text task failed",
                detail=message,
                completed_steps=3,
                active=False,
                model_name=payload.modelName,
            )
            raise HTTPException(status_code=400, detail=message)

        target_sr, audio_opt = output
        set_text_audio_progress(
            stage="finalizing",
            title="Write output",
            detail="Saving the generated result into the managed task directory.",
            completed_steps=4,
            model_name=payload.modelName,
        )
        output_path = text_output_path_in_directory(trimmed_text, output_root)
        save_audio(str(output_path), audio_opt, target_sr, f32=False, format="wav")
        set_text_audio_progress(
            stage="completed",
            title="Text task complete",
            detail="Generated speech and converted it into the selected voice model.",
            completed_steps=5,
            active=False,
            model_name=payload.modelName,
        )
        return {
            "message": message,
            "sourceAudioURL": preserved_source_path.as_uri() if preserved_source_path else None,
            "outputAudioURL": output_path.as_uri(),
            "outputDirectoryURL": output_root.as_uri(),
        }
    finally:
        if source_path is not None:
            shutil.rmtree(source_path.parent, ignore_errors=True)
        output = None
        reset_task_runtime("text generation", release_chattts=True)


def run_batch(payload: BatchInferencePayload) -> dict:
    output: tuple[int, np.ndarray] | None = None
    try:
        ensure_model_loaded(payload.modelName)
        output_root = Path(local_path(payload.outputDirectoryURL) or payload.outputDirectoryURL)
        output_root.mkdir(parents=True, exist_ok=True)

        input_paths: list[str] = []
        if payload.inputDirectoryURL:
            input_directory = Path(local_path(payload.inputDirectoryURL) or payload.inputDirectoryURL)
            if not input_directory.exists():
                raise HTTPException(status_code=400, detail="Input directory does not exist.")
            input_paths = [
                str(path)
                for path in sorted(input_directory.iterdir())
                if path.is_file()
            ]
        else:
            input_paths = [local_path(item) or item for item in payload.inputFileURLs]

        if not input_paths:
            raise HTTPException(status_code=400, detail="No batch input files were provided.")

        index_path = payload.indexPath or get_index_path_from_model(payload.modelName)
        messages: list[str] = []
        output_file_urls: list[str] = []

        for path in input_paths:
            with engine_lock:
                message, output = vc.vc_single(
                    payload.speakerId,
                    path,
                    payload.transpose,
                    None,
                    payload.f0Method,
                    "",
                    index_path or "",
                    payload.indexRate,
                    payload.filterRadius,
                    payload.resampleSR,
                    payload.rmsMixRate,
                    payload.protect,
                )
            if output is not None:
                target_sr, audio_opt = output
                output_name = f"{Path(path).name}.{payload.format}"
                output_path = output_root / output_name
                save_audio(
                    str(output_path),
                    audio_opt,
                    target_sr,
                    f32=False,
                    format=payload.format,
                )
                output_file_urls.append(output_path.as_uri())
            output = None
            messages.append(f"{Path(path).name} -> {message}")

        return {
            "message": "\n".join(messages),
            "outputDirectoryURL": output_root.as_uri(),
            "outputFileURLs": output_file_urls,
        }
    finally:
        output = None
        reset_task_runtime("batch conversion")


def model_file_path(model_name: str) -> Path:
    model_path = weight_root / model_name
    if not model_path.exists():
        raise HTTPException(status_code=400, detail="Selected model file does not exist.")
    return model_path


def resolved_index_path(model_name: str, index_path: str | None) -> str:
    resolved = index_path or get_index_path_from_model(model_name) or ""
    if not resolved:
        return ""
    if not Path(resolved).exists():
        if index_path:
            raise HTTPException(status_code=400, detail="Selected index file does not exist.")
        return ""
    return resolved


def realtime_status_payload() -> dict:
    controller = get_realtime_controller()
    return {
        "devices": controller.devices_snapshot(),
        "status": controller.status_snapshot(),
    }


def configure_realtime(payload: RealtimeConfigurePayload, allow_restart: bool = False) -> dict:
    try:
        return get_realtime_controller().configure(
            payload.model_dump(exclude_none=True),
            allow_restart=allow_restart,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Realtime configure failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def start_realtime(payload: RealtimeStartPayload) -> dict:
    try:
        model_name = payload.modelName
        ensure_model_loaded(model_name)
        index_path = resolved_index_path(model_name, payload.indexPath)
        return get_realtime_controller().start(
            payload.model_dump(exclude_none=True),
            str(model_file_path(model_name)),
            index_path,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Realtime start failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def stop_realtime() -> dict:
    try:
        return get_realtime_controller().stop()
    except Exception as exc:
        logger.exception("Realtime stop failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app = FastAPI(title="Swift RVC Phase 1 API")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "apiVersion": BACKEND_API_VERSION,
        "backendVersion": BACKEND_BUILD_VERSION,
        "sessionId": BACKEND_SESSION_ID,
        "sessionStartedAt": BACKEND_SESSION_STARTED_AT,
        "device": str(config.device),
        "selectedModel": selected_model_name,
        "realtimeRunning": get_realtime_controller().running if realtime_controller is not None else False,
    }


@app.get("/phase1/catalog")
def catalog() -> dict:
    index_paths = scan_indices()
    return {
        "speakerCount": 0,
        "models": [
            {
                "name": name,
                "indexPath": get_index_path_from_model(name) or "",
                "infoSummary": "",
            }
            for name in scan_models()
        ],
        "indexPaths": index_paths,
    }


@app.post("/phase1/select-model")
def select_model(payload: SelectModelRequest) -> dict:
    return ensure_model_loaded(payload.name)


@app.post("/phase1/unload-model")
def unload_selected_model() -> dict:
    # Keep the HTTP surface thin: the bridge only needs a compact success payload for model clearing.
    return unload_model()


@app.post("/phase1/convert-single")
def convert_single(payload: SingleInferencePayload) -> dict:
    return run_single(payload)


@app.post("/phase1/convert-text")
def convert_text(payload: TextAudioPayload) -> dict:
    try:
        return run_text(payload)
    except HTTPException as exc:
        if exc.detail:
            set_text_audio_progress(
                stage="failed",
                title="Text task failed",
                detail=str(exc.detail),
                completed_steps=max(get_text_audio_progress_snapshot().get("completedSteps", 0), 0),
                active=False,
                model_name=payload.modelName,
            )
        raise
    except Exception as exc:
        set_text_audio_progress(
            stage="failed",
            title="Text task failed",
            detail=str(exc),
            completed_steps=max(get_text_audio_progress_snapshot().get("completedSteps", 0), 0),
            active=False,
            model_name=payload.modelName,
        )
        raise


@app.get("/phase1/text-status")
def text_status() -> dict:
    return get_text_audio_progress_snapshot()


@app.post("/phase1/convert-batch")
def convert_batch(payload: BatchInferencePayload) -> dict:
    return run_batch(payload)


@app.post("/phase1/export-onnx")
def export_onnx(payload: ExportOnnxPayload) -> dict:
    return run_export_onnx(payload)


@app.post("/phase1/ckpt-compare")
def ckpt_compare(payload: CheckpointComparePayload) -> dict:
    return run_checkpoint_compare(payload)


@app.post("/phase1/ckpt-show")
def ckpt_show(payload: CheckpointInfoPayload) -> dict:
    return run_checkpoint_show(payload)


@app.post("/phase1/ckpt-modify")
def ckpt_modify(payload: CheckpointModifyPayload) -> dict:
    return run_checkpoint_modify(payload)


@app.post("/phase1/ckpt-merge")
def ckpt_merge(payload: CheckpointMergePayload) -> dict:
    return run_checkpoint_merge(payload)


@app.post("/phase1/ckpt-extract")
def ckpt_extract(payload: CheckpointExtractPayload) -> dict:
    return run_checkpoint_extract(payload)


@app.get("/phase1/uvr-models")
def uvr_models() -> dict:
    return {"modelNames": scan_uvr_models()}


@app.post("/phase1/uvr-convert")
def uvr_convert(payload: UVRConvertPayload) -> dict:
    return run_uvr(payload)


@app.post("/phase1/uvr-release")
def uvr_release() -> dict:
    return release_uvr_runtime()


@app.post("/phase1/release-runtime-memory")
def release_runtime_memory_route() -> dict:
    return release_runtime_memory()


@app.get("/phase1/assets-integrity")
def assets_integrity() -> dict:
    return asset_integrity_report()


@app.post("/phase1/assets-download")
def assets_download() -> dict:
    return download_assets()


@app.get("/phase1/realtime/devices")
def realtime_devices() -> dict:
    try:
        return get_realtime_controller().update_devices()
    except Exception as exc:
        logger.exception("Realtime device refresh failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/phase1/realtime/status")
def realtime_status() -> dict:
    return realtime_status_payload()


@app.post("/phase1/realtime/configure")
def realtime_configure(payload: RealtimeConfigurePayload) -> dict:
    return configure_realtime(payload, allow_restart=True)


@app.post("/phase1/realtime/start")
def realtime_start(payload: RealtimeStartPayload) -> dict:
    return start_realtime(payload)


@app.post("/phase1/realtime/stop")
def realtime_stop() -> dict:
    return stop_realtime()


if __name__ == "__main__":
    logger.info("Starting Phase 1 API on port %s", config.listen_port)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config.listen_port,
        log_level="info",
    )
