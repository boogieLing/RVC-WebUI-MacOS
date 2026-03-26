import traceback
from i18n.i18n import I18nAuto
from datetime import datetime
import torch

from .hash import model_hash_ckpt, hash_id, hash_similarity

i18n = I18nAuto()


def _render_model_info(cpt, short_id, long_id):
    return f"""{i18n("Model name")}: %s
{i18n("Sealing date")}: %s
{i18n("Model Author")}: %s
{i18n("Information")}: %s
{i18n("Sampling rate")}: %s
{i18n("Pitch guidance (f0)")}: %s
{i18n("Version")}: %s
{i18n("ID(short)")}: %s
{i18n("ID(long)")}: %s""" % (
        cpt.get("name", i18n("Unknown")),
        datetime.fromtimestamp(float(cpt.get("timestamp", 0))),
        cpt.get("author", i18n("Unknown")),
        cpt.get("info", i18n("None")),
        cpt.get("sr", i18n("Unknown")),
        i18n("Exist") if cpt.get("f0", 0) == 1 else i18n("Not exist"),
        cpt.get("version", i18n("None")),
        short_id,
        long_id,
    )


def show_model_info(cpt, show_long_id=False):
    try:
        if cpt.get("_auto_converted_training_checkpoint"):
            return _render_model_info(
                cpt,
                cpt.get("id", i18n("None")),
                i18n("Hidden") if not show_long_id else cpt.get("hash", i18n("None")),
            )
        h = model_hash_ckpt(cpt)
        id = hash_id(h)
        idread = cpt.get("id", "None")
        hread = cpt.get("hash", "None")
        if id != idread:
            id += (
                "("
                + i18n("Actually calculated")
                + "), "
                + idread
                + "("
                + i18n("Read from model")
                + ")"
            )
        sim = hash_similarity(h, hread)
        if not isinstance(sim, str):
            sim = "%.2f%%" % (sim * 100)
        if not show_long_id:
            h = i18n("Hidden")
            if h != hread:
                h = i18n("Similarity") + " " + sim + " -> " + h
        elif h != hread:
            h = (
                i18n("Similarity")
                + " "
                + sim
                + " -> "
                + h
                + "("
                + i18n("Actually calculated")
                + "), "
                + hread
                + "("
                + i18n("Read from model")
                + ")"
            )
        txt = _render_model_info(cpt, id, h)
    except (FileNotFoundError, OSError):
        idread = cpt.get("id", i18n("None"))
        hread = cpt.get("hash", i18n("None"))
        if not show_long_id and hread not in (None, "", i18n("None"), "None"):
            long_id = i18n("Read from model")
        else:
            long_id = hread
        txt = _render_model_info(cpt, idread, long_id)
    except:
        txt = traceback.format_exc()

    return txt


def show_info(path):
    try:
        if hasattr(path, "name"):
            path = path.name
        a = torch.load(path, map_location="cpu")
        txt = show_model_info(a, show_long_id=True)
        del a
    except:
        txt = traceback.format_exc()

    return txt
