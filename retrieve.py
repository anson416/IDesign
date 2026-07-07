"""Text -> 3D-asset retrieval via OpenShape + CLIP.

This module is import-safe: importing it does NOT load torch / CLIP / the
embedding bank. Heavy resources are loaded lazily on first retrieval and
cached on a module-level singleton (`_BACKEND`), so callers that only need
the scene-graph generation path (no assets) pay nothing.

Public API
----------
load_backend(force=False) -> RetrievalBackend
    Explicitly load CLIP + the OpenShape embedding bank. Raises if the deps
    are missing; returns the cached backend if already loaded.

retrieve_candidates(text, top, sim_th, worst=False, filter_fn=None) -> list[dict]
    Return up to `top` candidate assets for a text query, sorted best-first by
    CLIP cosine similarity (or worst-first if worst=True).

retrieve_asset_for_object(obj, assets_dir, match="best", sim_th=0.1) -> Optional[str]
    Download the .glb for one scene-graph object into `assets_dir/<id>.glb`.
    match="best" keeps the top-ranked candidate; match="worst" picks the
    lowest-similarity candidate that still passes the filter. Returns the
    objaverse uid, or None if retrieval/download failed.

retrieve_scene_assets(scene_graph, assets_dir, match="best", sim_th=0.1,
                     verbose=False) -> dict
    Iterate the real objects in a scene_graph and retrieve an asset for each.
    Returns {"placed": [ids], "skipped": [ids], "uids": {id: uid}}.

When run as a script (`python retrieve.py`) it reproduces the original
behaviour: read scene_graph.json from the CWD, retrieve the BEST asset per
object into ./Assets/.
"""

import json
import os
import re
import shutil
import threading
from typing import Optional

# objaverse writes a large cache; redirect it off the home quota when possible.
_OBJV_BASE = os.environ.get(
    "VLMUNR_OBJAVERSE_BASE", "/research/d2/fyp24/yflam1/.objaverse_cache"
)
try:
    import objaverse  # noqa: F401  (import for its side effect of setting BASE_PATH)

    os.makedirs(_OBJV_BASE, exist_ok=True)
    objaverse.BASE_PATH = _OBJV_BASE
    objaverse._VERSIONED_PATH = os.path.join(_OBJV_BASE, "hf-objaverse-v1")
except Exception:
    objaverse = None  # type: ignore


def preprocess(input_string: str) -> str:
    """Strip digits and underscores, mirroring the original retrieval query
    preprocessing (`chair_1` -> `chair`)."""
    wo_numericals = re.sub(r"\d", "", input_string)
    return wo_numericals.replace("_", " ")


def get_filter_fn():
    """Default objaverse asset filter (animation/face count bounds)."""
    face_min, face_max = 0, 34985808
    anim_min, anim_max = 0, 563
    anim_n = not (anim_min > 0 or anim_max < 563)
    face_n = not (face_min > 0 or face_max < 34985808)
    return lambda x: (
        (anim_n or anim_min <= x["anims"] <= anim_max)
        and (face_n or face_min <= x["faces"] <= face_max)
    )


class RetrievalBackend:
    """Lazy, cached holder of the CLIP model + OpenShape embedding bank."""

    def __init__(self):
        self.clip_model = None
        self.clip_prep = None
        self.us = None  # objaverse uids, aligned with feats rows
        self.feats = None  # (N, D) embedding tensor
        self.meta = None  # uid -> metadata dict
        self._device = "cpu"

    @property
    def available(self) -> bool:
        return self.clip_model is not None and self.feats is not None

    def load(self):
        """Load torch + CLIP + the embedding bank. Idempotent."""
        if self.available:
            return self
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from huggingface_hub import hf_hub_download
        import transformers

        global objaverse
        if objaverse is None:
            import objaverse  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        print(
            "Device: ",
            torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        )

        # Pre-computed embeddings (downloaded into ./OpenShape-Embeddings).
        emb_dir = "OpenShape-Embeddings"
        meta = json.load(
            open(
                hf_hub_download(
                    "OpenShape/openshape-objaverse-embeddings",
                    "objaverse_meta.json",
                    token=True,
                    repo_type="dataset",
                    local_dir=emb_dir,
                )
            )
        )
        self.meta = {x["u"]: x for x in meta["entries"]}
        deser = torch.load(
            hf_hub_download(
                "OpenShape/openshape-objaverse-embeddings",
                "objaverse.pt",
                token=True,
                repo_type="dataset",
                local_dir=emb_dir,
            ),
            map_location="cpu",
        )
        self.us = deser["us"]
        self.feats = deser["feats"]

        half = torch.float16 if torch.cuda.is_available() else torch.bfloat16
        self.clip_model, self.clip_prep = self._load_openclip(transformers, half, device)
        torch.set_grad_enabled(False)
        return self

    def _load_openclip(self, transformers, half, device):
        print("Locking...")
        sys = __import__("sys")
        sys.clip_move_lock = threading.Lock()
        print("Locked.")
        clip_model, clip_prep = (
            transformers.CLIPModel.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                low_cpu_mem_usage=True,
                torch_dtype=half,
                offload_state_dict=True,
            ),
            transformers.CLIPProcessor.from_pretrained(
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
            ),
        )
        if device == "cuda":
            with sys.clip_move_lock:
                clip_model.cuda()
        return clip_model, clip_prep

    def encode_text(self, text: str):
        import torch  # noqa: F401
        from torch.nn import functional as F

        device = self.clip_model.device
        tn = self.clip_prep(
            text=[text], return_tensors="pt", truncation=True, max_length=76
        ).to(device)
        enc_raw = self.clip_model.get_text_features(**tn)
        enc = (
            enc_raw.pooler_output if hasattr(enc_raw, "pooler_output") else enc_raw
        ).float().cpu()
        return enc

    def retrieve(self, embedding, top, sim_th=0.0, filter_fn=None, worst=False):
        """Return up to `top` candidate dicts sorted by similarity.

        worst=False -> best (highest similarity) first (the original behaviour).
        worst=True  -> worst (lowest similarity) first.
        """
        import torch
        from torch.nn import functional as F

        sims = []
        embedding = F.normalize(embedding.detach().cpu(), dim=-1).squeeze()
        for chunk in torch.split(self.feats, 10240):
            sims.append(embedding @ F.normalize(chunk.float(), dim=-1).T)
        sims = torch.cat(sims)
        # Sort descending so idx aligns best-first regardless of mode.
        sims, idx = torch.sort(sims, descending=True)
        sim_mask = sims > sim_th
        sims = sims[sim_mask]
        idx = idx[sim_mask]
        results = []
        for i, sim in zip(idx, sims):
            if self.us[i] in self.meta:
                if filter_fn is None or filter_fn(self.meta[self.us[i]]):
                    results.append(dict(self.meta[self.us[i]], sim=sim))
        if worst:
            results = list(reversed(results))
        return results[:top]


# Module-level singleton; loaded on demand.
_BACKEND: Optional[RetrievalBackend] = None


def load_backend(force: bool = False) -> RetrievalBackend:
    """Load (or return the cached) retrieval backend.

    Raises RuntimeError if torch / transformers / huggingface assets are
    unavailable. Callers that can tolerate absence should catch and degrade.
    """
    global _BACKEND
    if _BACKEND is None or force or not _BACKEND.available:
        backend = RetrievalBackend()
        backend.load()
        _BACKEND = backend
    return _BACKEND


def backend_available() -> bool:
    """True if the backend is already loaded (does not attempt to load)."""
    return _BACKEND is not None and _BACKEND.available


def retrieve_candidates(
    text: str,
    top: int = 1,
    sim_th: float = 0.1,
    worst: bool = False,
    filter_fn=None,
):
    """Retrieve candidate asset dicts for a text query."""
    backend = load_backend()
    enc = backend.encode_text(text)
    return backend.retrieve(enc, top=top, sim_th=sim_th, filter_fn=filter_fn, worst=worst)


def _object_query_text(obj: dict) -> Optional[str]:
    """Build the CLIP text query for a scene-graph object, mirroring the
    original retrieve.py query. Returns None for objects without style/
    material (e.g. room priors), which are skipped."""
    if "style" not in obj or "material" not in obj:
        return None
    style, material = obj["style"], obj["material"]
    return (
        preprocess("A high-poly " + obj["new_object_id"])
        + f" with {material} material and in {style} style, high quality"
    )


def retrieve_asset_for_object(
    obj: dict,
    assets_dir: str,
    match: str = "best",
    sim_th: float = 0.1,
    download: bool = True,
    autoload: bool = False,
) -> Optional[str]:
    """Retrieve and (optionally) download a .glb asset for one object.

    match: "best" (highest CLIP sim) or "worst" (lowest CLIP sim passing the
    filter). Returns the objaverse uid on success, else None. The .glb is
    written to `<assets_dir>/<new_object_id>.glb` when download=True.

    autoload: if False (default) the function returns None when the retrieval
    backend is NOT already loaded, rather than implicitly triggering a ~3GB
    CLIP+embedding download. Callers that want retrieval must call
    load_backend() first (the scene CLI does this explicitly). Set autoload=True
    to load-on-demand (e.g. for the standalone `python retrieve.py` script).
    """
    if match not in ("best", "worst"):
        raise ValueError(f"Unknown match mode: {match!r}")
    text = _object_query_text(obj)
    if text is None:
        return None
    obj_id = obj["new_object_id"]

    if not autoload and not backend_available():
        # Graceful no-op: do not implicitly download the embedding bank.
        return None

    try:
        candidates = retrieve_candidates(
            text, top=1, sim_th=sim_th, worst=(match == "worst"),
            filter_fn=get_filter_fn(),
        )
    except Exception as e:
        print(f"[retrieve] {obj_id}: retrieval failed: {e}")
        return None
    if not candidates:
        print(f"[retrieve] {obj_id}: no candidate cleared sim_th={sim_th}")
        return None

    chosen = candidates[0]
    uid = chosen["u"]
    if not download:
        return uid

    try:
        import multiprocessing

        processes = multiprocessing.cpu_count()
        objaverse_objects = objaverse.load_objects(
            uids=[uid], download_processes=processes
        )
    except Exception as e:
        print(f"[retrieve] {obj_id}: asset download failed: {e}")
        return uid  # uid known, but .glb not on disk
    os.makedirs(assets_dir, exist_ok=True)
    _move_files(objaverse_objects, assets_dir, obj_id)
    return uid


def _move_files(file_dict, destination_folder, obj_id):
    for item_id, file_path in file_dict.items():
        destination_path = os.path.join(destination_folder, f"{obj_id}.glb")
        shutil.move(file_path, destination_path)
        print(f"File {item_id} moved from {file_path} to {destination_path}")


# ----------------------------------------------------------------------------
# Scene-graph-level helpers (room-prior aware).
# ----------------------------------------------------------------------------

ROOM_PRIOR_IDS = frozenset(
    {
        "south_wall",
        "north_wall",
        "east_wall",
        "west_wall",
        "middle of the room",
        "ceiling",
    }
)


def is_real_object(item: dict) -> bool:
    if "itemType" in item:
        return False
    obj_id = item.get("new_object_id")
    if obj_id is None:
        return False
    return obj_id not in ROOM_PRIOR_IDS


def retrieve_scene_assets(
    scene_graph,
    assets_dir: str,
    match: str = "best",
    sim_th: float = 0.1,
    verbose: bool = False,
    autoload: bool = True,
) -> dict:
    """Retrieve an asset for every real object in a scene_graph.

    scene_graph may be either the flat list (after backtrack()) or the dict
    {"objects_in_room": [...]} form (after create_initial_design).

    autoload: passed through to retrieve_asset_for_object. Defaults to True so
    the standalone `python retrieve.py` script (which calls this) loads the
    backend on demand. The scene CLI loads the backend explicitly first, then
    passes autoload=False to avoid a redundant reload.

    Returns {"placed": [...], "skipped": [...], "uids": {id: uid}}.
    """
    if isinstance(scene_graph, dict) and "objects_in_room" in scene_graph:
        objects = scene_graph["objects_in_room"]
    else:
        objects = scene_graph

    placed, skipped, uids = [], [], {}
    for obj in objects:
        if not isinstance(obj, dict) or not is_real_object(obj):
            continue
        uid = retrieve_asset_for_object(
            obj, assets_dir, match=match, sim_th=sim_th, autoload=autoload
        )
        if uid is not None:
            placed.append(obj["new_object_id"])
            uids[obj["new_object_id"]] = uid
        else:
            skipped.append(obj["new_object_id"])
        if verbose:
            print(f"[retrieve] {obj['new_object_id']}: {uid}")
    return {"placed": placed, "skipped": skipped, "uids": uids}


# ----------------------------------------------------------------------------
# Script entry point (reproduces original `python retrieve.py` behaviour).
# ----------------------------------------------------------------------------

def _main():
    file_path = "scene_graph.json"
    with open(file_path, "r") as file:
        scene_graph = json.load(file)
    retrieve_scene_assets(
        scene_graph, os.path.join(os.getcwd(), "Assets"), match="best",
        sim_th=0.1, verbose=True,
    )


if __name__ == "__main__":
    _main()
