from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path

import torch

from .layers.synthesizers import SynthesizerTrnMsNSFsid
from .jit import load_inputs, export_jit_model, save_pickle


def validate_inference_checkpoint(cpt: OrderedDict) -> None:
    if "weight" in cpt and "config" in cpt:
        return

    if "model" in cpt and "config" not in cpt:
        raise ValueError(
            "This .pth is a training checkpoint, not an extracted inference model. "
            "Use CKPT -> EXTRACT to generate a deployable weights file first."
        )

    missing = [key for key in ("weight", "config") if key not in cpt]
    raise ValueError(
        "Unsupported model checkpoint format. Missing required field(s): "
        + ", ".join(missing)
    )


def _derive_model_version(config_data: dict) -> str:
    ssl_dim = int(config_data.get("model", {}).get("ssl_dim", 256) or 256)
    return "v2" if ssl_dim >= 768 else "v1"


def _derive_use_f0(weight: OrderedDict) -> int:
    return int(
        "enc_p.f0_emb.weight" in weight
        or any(key.startswith("f0_decoder.") for key in weight)
    )


def _has_unsupported_custom_architecture(weight: OrderedDict) -> bool:
    return "emb_uv.weight" in weight or any(
        key.startswith("f0_decoder.") for key in weight
    )


def _build_config_list(config_data: dict) -> list:
    data = config_data.get("data", {})
    model = config_data.get("model", {})
    return [
        int(data["filter_length"]) // 2 + 1,
        32,
        int(model["inter_channels"]),
        int(model["hidden_channels"]),
        int(model["filter_channels"]),
        int(model["n_heads"]),
        int(model["n_layers"]),
        int(model["kernel_size"]),
        float(model["p_dropout"]),
        str(model["resblock"]),
        list(model["resblock_kernel_sizes"]),
        list(model["resblock_dilation_sizes"]),
        list(model["upsample_rates"]),
        int(model["upsample_initial_channel"]),
        list(model["upsample_kernel_sizes"]),
        int(model["n_speakers"]),
        int(model["gin_channels"]),
        int(data["sampling_rate"]),
    ]


def _infer_harmonic_num(weight: OrderedDict, fallback: int = 0) -> int:
    tensor = weight.get("dec.m_source.l_linear.weight")
    if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) >= 2:
        return max(int(tensor.shape[1]) - 1, 0)
    return fallback


def _infer_flow_n_layers(weight: OrderedDict, fallback: int = 3) -> int:
    prefix = "flow.flows.0.enc.in_layers."
    indices = set()
    for key in weight:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        layer_id = remainder.split(".", 1)[0]
        if layer_id.isdigit():
            indices.add(int(layer_id))
    return max(indices) + 1 if indices else fallback


def _convert_training_checkpoint(
    cpt: OrderedDict, pth_path: str | BytesIO
) -> OrderedDict:
    if not isinstance(pth_path, str):
        raise ValueError(
            "This .pth is a training checkpoint and needs a sidecar .config.json file on disk."
        )

    config_path = Path(pth_path).with_suffix(".config.json")
    if not config_path.exists():
        raise ValueError(
            "This .pth is a training checkpoint, not an extracted inference model. "
            "Add a matching .config.json or use CKPT -> EXTRACT first."
        )

    config_data = json.loads(config_path.read_text())
    weight = cpt.get("model")
    if not isinstance(weight, dict):
        raise ValueError("Training checkpoint is missing the model weight block.")
    if _has_unsupported_custom_architecture(weight):
        raise ValueError(
            "This training checkpoint uses a custom voice-conversion architecture "
            "(for example emb_uv/f0_decoder) that is not compatible with the official "
            "RVC inference standard used by this app. Re-export it as an official "
            "RVC small model, or use an official-compatible checkpoint instead."
        )

    converted = OrderedDict()
    converted["weight"] = OrderedDict(
        (key, value.half() if hasattr(value, "half") else value)
        for key, value in weight.items()
        if "enc_q" not in key
    )
    config_list = _build_config_list(config_data)
    config_list[15] = int(weight["emb_g.weight"].shape[0])
    config_list[16] = int(weight["emb_g.weight"].shape[1])
    converted["config"] = config_list
    converted["f0"] = _derive_use_f0(weight)
    converted["version"] = _derive_model_version(config_data)
    converted["harmonic_num"] = _infer_harmonic_num(weight)
    converted["flow_n_layers"] = _infer_flow_n_layers(weight)
    converted["name"] = Path(pth_path).stem
    converted["sr"] = int(config_data.get("data", {}).get("sampling_rate", 0) or 0)
    converted["info"] = "Auto-converted from training checkpoint with sidecar config."
    converted["author"] = config_data.get("author", "Unknown")
    converted["_auto_converted_training_checkpoint"] = True
    return converted


def get_synthesizer(cpt: OrderedDict, device=torch.device("cpu")):
    validate_inference_checkpoint(cpt)
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        encoder_dim = 256
    elif version == "v2":
        encoder_dim = 768
    net_g = SynthesizerTrnMsNSFsid(
        *cpt["config"],
        encoder_dim=encoder_dim,
        use_f0=if_f0 == 1,
        harmonic_num=int(cpt.get("harmonic_num", 0) or 0),
        flow_n_layers=int(cpt.get("flow_n_layers", 3) or 3),
    )
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.float()
    net_g.eval().to(device)
    net_g.remove_weight_norm()
    return net_g, cpt


def load_synthesizer(
    pth_path: str | BytesIO, device=torch.device("cpu")
):
    cpt = torch.load(pth_path, map_location=torch.device("cpu"), weights_only=True)
    if "weight" not in cpt and "model" in cpt:
        cpt = _convert_training_checkpoint(cpt, pth_path)
    return get_synthesizer(cpt, device)


def synthesizer_jit_export(
    model_path: str,
    mode: str = "script",
    inputs_path: str = None,
    save_path: str = None,
    device=torch.device("cpu"),
    is_half=False,
):
    if not save_path:
        save_path = model_path.rstrip(".pth")
        save_path += ".half.jit" if is_half else ".jit"
    if "cuda" in str(device) and ":" not in str(device):
        device = torch.device("cuda:0")
    from rvc.synthesizer import load_synthesizer

    model, cpt = load_synthesizer(model_path, device)
    assert isinstance(cpt, dict)
    model.forward = model.infer
    inputs = None
    if mode == "trace":
        inputs = load_inputs(inputs_path, device, is_half)
    ckpt = export_jit_model(model, mode, inputs, device, is_half)
    cpt.pop("weight")
    cpt["model"] = ckpt["model"]
    cpt["device"] = device
    save_pickle(cpt, save_path)
    return cpt
