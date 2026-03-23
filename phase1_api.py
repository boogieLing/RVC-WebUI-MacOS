#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import sys
import threading
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
from infer.modules.vc import VC
from infer.modules.vc.info import show_model_info
from infer.modules.vc.utils import get_index_path_from_model
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


class SelectModelRequest(BaseModel):
    name: str


class SingleInferencePayload(BaseModel):
    modelName: str
    inputFileURL: str
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
    transpose: float
    f0Method: str
    indexPath: Optional[str] = None
    indexRate: float
    filterRadius: float
    resampleSR: float
    rmsMixRate: float
    protect: float


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
    return {
        "modelName": model_name,
        "modelInfoSummary": model_state[5] if len(model_state) > 5 else "",
        "indexPaths": [item for item in scan_indices() if item],
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
            0,
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
                0,
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
        raise HTTPException(status_code=400, detail="Index path is required for realtime conversion.")
    if not Path(resolved).exists():
        raise HTTPException(status_code=400, detail="Selected index file does not exist.")
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


@app.post("/phase1/convert-single")
def convert_single(payload: SingleInferencePayload) -> dict:
    return run_single(payload)


@app.post("/phase1/convert-batch")
def convert_batch(payload: BatchInferencePayload) -> dict:
    return run_batch(payload)


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
