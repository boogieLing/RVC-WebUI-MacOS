import os, pathlib
from contextlib import contextmanager

from fairseq import checkpoint_utils
import torch


def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                str(pathlib.Path(root, name))
                for path in [os.getenv("outside_index_root"), os.getenv("index_root")]
                for root, _, files in os.walk(path, topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


@contextmanager
def torch_load_weights_compat():
    original_torch_load = torch.load

    def compat_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compat_torch_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def load_hubert(device, is_half):
    with torch_load_weights_compat():
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["assets/hubert/hubert_base.pt"],
            suffix="",
        )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
