#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
load_dotenv("sha256.env")

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
from infer.modules.uvr5.modules import uvr
from rvc.onnx import export_onnx as export_rvc_onnx
from realtime_vc import RealtimeVCController


logger = logging.getLogger("phase1_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

ROOT = Path(now_dir)
TMP_DIR = ROOT / "TEMP"
LOG_DIR = ROOT / "logs"
OUTPUT_DIR = ROOT / "opt"
for directory in (TMP_DIR, LOG_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

config = Config()
vc = VC(config)
engine_lock = threading.RLock()
selected_model_name: str = ""
realtime_controller: RealtimeVCController | None = None

weight_root = Path(os.getenv("weight_root", "assets/weights"))
index_root = Path(os.getenv("index_root", "logs"))
outside_index_root = Path(os.getenv("outside_index_root", "assets/indices"))
weight_uvr5_root = Path(os.getenv("weight_uvr5_root", "assets/uvr5_weights"))


def _sanitize_model_info_summary(value: str | None) -> tuple[str, str | None]:
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
    return summary, None


class SelectModelRequest(BaseModel):
    name: str


class SingleInferencePayload(BaseModel):
    modelName: str
    inputFileURL: str
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
    model_info_summary, model_info_error = _sanitize_model_info_summary(model_state[5] if len(model_state) > 5 else "")
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
        "modelName": payload.modelName,
        "stagedInputCount": len(staged_paths),
    }


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


def run_single(payload: SingleInferencePayload) -> dict:
    ensure_model_loaded(payload.modelName)
    input_path = local_path(payload.inputFileURL)
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Input audio file does not exist.")

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
    output_path = single_output_path(input_path)
    save_audio(str(output_path), audio_opt, target_sr, f32=True, format="wav")
    return {
        "message": message,
        "outputAudioURL": output_path.as_uri(),
    }


def run_batch(payload: BatchInferencePayload) -> dict:
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
            save_audio(
                str(output_root / output_name),
                audio_opt,
                target_sr,
                f32=True,
                format=payload.format,
            )
        messages.append(f"{Path(path).name} -> {message}")

    return {
        "message": "\n".join(messages),
        "outputDirectoryURL": output_root.as_uri(),
    }


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
