import os
import traceback
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

from infer.lib.audio import resample_audio, get_audio_properties
import torch

from configs import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre

config = Config()


def release_uvr_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Executed torch.cuda.empty_cache()")
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.info("Executed torch.mps.empty_cache()")
    return {"released": True}


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    temp_root = os.getenv("TEMP") or os.getenv("TMP") or os.getenv("TMPDIR") or tempfile.gettempdir()
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            pre_fun = AudioPre(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                channels, rate = get_audio_properties(inp_path)

                # Check the audio stream's properties
                if channels == 2 and rate == 44100:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                    need_reformat = 0
                    done = 1
            except Exception as e:
                need_reformat = 1
                logger.warning(f"Exception {e} occured. Will reformat")
            if need_reformat == 1:
                original_name = Path(inp_path).stem or Path(inp_path).name
                tmp_path = os.path.join(
                    temp_root,
                    f"{original_name}.reformatted.wav",
                )
                resample_audio(inp_path, tmp_path, "pcm_s16le", "s16", 44100, "stereo")
                try:  # Remove the original file
                    os.remove(inp_path)
                except Exception as e:
                    print(f"Failed to remove the original file: {e}")
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        release_uvr_memory()
    yield "\n".join(infos)
