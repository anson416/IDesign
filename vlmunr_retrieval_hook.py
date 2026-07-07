"""Retrieval hook for the VLM-unreliability content-variant generators.

`vlmunr_variants.build_alt_scene` / `build_subst_scene` import
`retrieve_worst_match` / `retrieve_substitute` from this module. The previous
code referenced a module that did not exist in the repo, so worst-object and
substitution variants always silently degraded to "copy unchanged". This hook
routes them through the real OpenShape/CLIP retrieval in `retrieve.py`.

The heavy backend (torch / CLIP / embedding bank) is loaded lazily inside
`retrieve.py`; if it is unavailable (no GPU, no downloaded embeddings), every
function here returns None and the caller degrades gracefully.
"""

from typing import Optional

# Lazily import retrieve at call time so importing this module is cheap and
# never crashes on a missing optional dependency.


def _backend_ready() -> bool:
    """True only if the heavy retrieval backend is ALREADY loaded.

    We intentionally do NOT call load_backend() here: doing so would silently
    download the ~3GB OpenShape embedding bank + CLIP weights from inside a
    content-variant generator. The hook represents best-effort retrieval that
    degrades gracefully when the backend is not already up. Callers that want
    real retrieval (e.g. the scene CLI) must explicitly call
    `retrieve.load_backend()` first; if they don't, every hook function below
    returns None and the variant degrades (asset copied / intent recorded).
    """
    try:
        import retrieve

        return retrieve.backend_available()
    except Exception:
        return False


def retrieve_worst_match(
    object_id: str, rank: int, scene_dir: str
) -> Optional[str]:
    """Return the objaverse uid of the rank-th WORST-CLIP-matching asset for
    `object_id`, or None if retrieval is unavailable.

    `rank` selects how far from the worst end to go (0 == absolute worst).
    The base scene's scene_graph.json + Assets/ in `scene_dir` are used to
    look up the object's style/material for the CLIP query.
    """
    try:
        import json as _json
        import os as _os

        import retrieve  # lazy

        if not _backend_ready():
            return None  # graceful: variant copies base asset / records intent
        scene_path = _os.path.join(scene_dir, "scene_graph.json")
        with open(scene_path) as f:
            scene_graph = _json.load(f)
        if isinstance(scene_graph, dict) and "objects_in_room" in scene_graph:
            objects = scene_graph["objects_in_room"]
        else:
            objects = scene_graph
        obj = next(
            (o for o in objects if isinstance(o, dict) and o.get("new_object_id") == object_id),
            None,
        )
        if obj is None:
            return None
        # rank=0 -> worst, rank=2 -> 3rd-from-worst, etc. top = rank+1.
        candidates = retrieve.retrieve_candidates(
            retrieve._object_query_text(obj) or "",
            top=rank + 1,
            sim_th=0.0,
            worst=True,
            filter_fn=retrieve.get_filter_fn(),
        )
        if len(candidates) <= rank:
            return None
        return candidates[rank]["u"]
    except Exception as e:  # retrieval unavailable / object lacks style
        print(f"[retrieval_hook] worst_match {object_id}: {e}")
        return None


def retrieve_substitute(
    object_id: str,
    mode: str,
    target_category: Optional[str],
    scene_dir: str,
) -> Optional[str]:
    """Return a substitute asset uid for `object_id`, or None.

    mode "within": a different instance of the SAME category (any candidate
        except the one already used by the base scene, when distinguishable).
    mode "cross": an instance of `target_category` (a different category).

    For "cross", the query text is built from the target category name with a
    generic style/material so CLIP retrieves a representative asset.
    """
    try:
        import retrieve

        if not _backend_ready():
            return None  # graceful: variant copies base asset / records intent
        if mode == "cross":
            if not target_category:
                return None
            # Generic query for the target category.
            text = retrieve.preprocess("A high-poly " + target_category) + (
                " with wood material and in modern style, high quality"
            )
            candidates = retrieve.retrieve_candidates(
                text, top=1, sim_th=0.0, worst=False,
                filter_fn=retrieve.get_filter_fn(),
            )
            return candidates[0]["u"] if candidates else None

        # within: reuse the object's own query but skip the top match (rank 1
        # best -> the 2nd-best instance of the same category).
        import json as _json
        import os as _os

        scene_path = _os.path.join(scene_dir, "scene_graph.json")
        with open(scene_path) as f:
            scene_graph = _json.load(f)
        if isinstance(scene_graph, dict) and "objects_in_room" in scene_graph:
            objects = scene_graph["objects_in_room"]
        else:
            objects = scene_graph
        obj = next(
            (o for o in objects if isinstance(o, dict) and o.get("new_object_id") == object_id),
            None,
        )
        if obj is None:
            return None
        candidates = retrieve.retrieve_candidates(
            retrieve._object_query_text(obj) or "",
            top=2,
            sim_th=0.0,
            worst=False,
            filter_fn=retrieve.get_filter_fn(),
        )
        if len(candidates) >= 2:
            return candidates[1]["u"]
        if candidates:
            return candidates[0]["u"]
        return None
    except Exception as e:
        print(f"[retrieval_hook] subst {object_id}/{mode}: {e}")
        return None
