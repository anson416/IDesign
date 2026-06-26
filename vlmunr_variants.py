"""Content-variant generator for the VLM-unreliability audit.

Produces six variant scenes from an I-Design `scene_graph.json`:

  Removal variants (seeded, deterministic):
    variant_half    -> keep round(n/2) real objects
    variant_quarter -> keep round(n/4) real objects
    variant_eighth  -> keep round(n/8) real objects
  (always keeps at least one object)

  Worst-match variants (structured around a retrieval hook):
    variant_alt_0   -> swap each asset for the rank-0 (worst-ranked) match
    variant_alt_2   -> swap each asset for the rank-2 match (from worst end)
    variant_alt_4   -> swap each asset for the rank-4 match (from worst end)

  The worst-match generator lazy-imports a retrieval hook (which itself
  lazy-imports torch / CLIP). When retrieval is unavailable (no GPU, no
  downloaded assets/models), it degrades gracefully: it copies the scene
  unchanged and records the intended substitution under the
  `_vlmunr_alt_intent` key so the rest of the pipeline and the tests still
  pass.

The room-prior filter and removal logic are exposed as pure, importable,
unit-testable functions.
"""

import argparse
import copy
import json
import os
import random
from typing import Optional

# Room-prior ids in the flat scene_graph list that are NOT real objects.
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

REMOVAL_VARIANTS = {
    "variant_half": 2,
    "variant_quarter": 4,
    "variant_eighth": 8,
}

ALT_VARIANTS = {
    "variant_alt_0": 0,
    "variant_alt_2": 2,
    "variant_alt_4": 4,
}


def is_real_object(item: dict) -> bool:
    """True if a scene_graph entry is a real placeable object (not a room
    prior).

    A real object has a `new_object_id` and no `itemType` field, and its id is
    not one of the known room-prior ids.
    """
    if "itemType" in item:
        return False
    obj_id = item.get("new_object_id")
    if obj_id is None:
        return False
    return obj_id not in ROOM_PRIOR_IDS


def filter_real_objects(scene_graph: list) -> list:
    """Return only the real objects from a flat scene_graph list."""
    return [item for item in scene_graph if is_real_object(item)]


def keep_count(n: int, divisor: int) -> int:
    """How many objects to keep for a removal fraction.

    Uses round(n / divisor), clamped to at least 1 (and at most n).
    """
    if n <= 0:
        return 0
    k = round(n / divisor)
    return max(1, min(k, n))


def select_kept_objects(
    real_objects: list, divisor: int, seed: int
) -> list:
    """Deterministically select the kept subset of objects for a removal
    fraction.

    Order of the returned list follows the original `real_objects` order so the
    output is stable and easy to diff.
    """
    n = len(real_objects)
    k = keep_count(n, divisor)
    rng = random.Random(seed)
    chosen = rng.sample(range(n), k)
    chosen_set = set(chosen)
    return [obj for i, obj in enumerate(real_objects) if i in chosen_set]


def build_removal_scene(real_objects: list, divisor: int, seed: int) -> list:
    """Build a variant scene_graph (a list) for a removal fraction."""
    return [copy.deepcopy(o) for o in select_kept_objects(real_objects, divisor, seed)]


def _try_retrieval_alt(object_id: str, rank: int, scene_dir: str) -> Optional[str]:
    """Lazy-imported retrieval hook returning a low-CLIP-ranked alternate asset
    uid for an object, or None if retrieval is unavailable.

    This intentionally imports heavy dependencies (torch / CLIP / objaverse)
    only when called, and returns None on any failure so the caller can degrade
    gracefully. It is NOT exercised in unit tests / smoke runs (no GPU, no
    downloads).
    """
    try:  # pragma: no cover - requires GPU + model downloads
        import torch  # noqa: F401  (lazy, import-light at module level)
        from vlmunr_retrieval_hook import retrieve_worst_match  # type: ignore

        return retrieve_worst_match(object_id, rank, scene_dir)
    except Exception:
        return None


def build_alt_scene(
    real_objects: list, rank: int, scene_dir: str
) -> tuple[list, dict]:
    """Build a worst-match variant scene_graph.

    Returns (scene_graph_list, intent_map). Each object's asset is swapped for a
    low-ranked retrieval result when the retrieval hook is available; otherwise
    the scene is copied unchanged and the intent is recorded per object id.
    """
    out = [copy.deepcopy(o) for o in real_objects]
    intent: dict[str, int] = {}
    for obj in out:
        obj_id = obj["new_object_id"]
        alt_uid = _try_retrieval_alt(obj_id, rank, scene_dir)
        if alt_uid is not None:
            obj["_vlmunr_alt_asset"] = alt_uid
        else:
            intent[obj_id] = rank
    return out, intent


def _write_scene(out_dir: str, scene_graph: list, assets_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scene_graph.json"), "w") as f:
        json.dump(scene_graph, f, indent=2)
    # Record where the renderer should read Assets from (the original scene).
    with open(os.path.join(out_dir, "vlmunr_assets_dir.txt"), "w") as f:
        f.write(assets_dir)


def generate_variants(scene_dir: str, seed: int = 42) -> dict:
    """Generate all six variant directories as siblings of `scene_dir`.

    Returns a mapping {variant_name: variant_dir}.
    """
    scene_path = os.path.join(scene_dir, "scene_graph.json")
    with open(scene_path) as f:
        scene_graph = json.load(f)
    real_objects = filter_real_objects(scene_graph)

    parent = os.path.dirname(os.path.abspath(scene_dir.rstrip(os.sep)))
    base_name = os.path.basename(os.path.abspath(scene_dir.rstrip(os.sep)))
    assets_dir = os.path.join(os.path.abspath(scene_dir), "Assets")

    created: dict[str, str] = {}

    for name, divisor in REMOVAL_VARIANTS.items():
        variant_scene = build_removal_scene(real_objects, divisor, seed)
        out_dir = os.path.join(parent, f"{base_name}_{name}")
        _write_scene(out_dir, variant_scene, assets_dir)
        created[name] = out_dir

    for name, rank in ALT_VARIANTS.items():
        variant_scene, intent = build_alt_scene(real_objects, rank, scene_dir)
        out_dir = os.path.join(parent, f"{base_name}_{name}")
        # Embed intent as a leading metadata entry so the file remains a single
        # JSON document; the renderer/filter ignore non-real entries.
        payload: list = list(variant_scene)
        if intent:
            payload = [{"_vlmunr_alt_intent": intent}] + payload
        _write_scene(out_dir, payload, assets_dir)
        created[name] = out_dir

    return created


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate content variants from an I-Design scene_graph.json"
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing scene_graph.json and Assets/",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    created = generate_variants(args.scene_dir, seed=args.seed)
    for name, path in created.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
