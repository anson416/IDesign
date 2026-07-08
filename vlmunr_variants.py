"""Content-variant generator for the VLM-unreliability audit.

Produces variant scenes from an I-Design `scene_graph.json`:

  Removal variants (seeded, deterministic):
    variant_half    -> keep round(n/2) real objects
    variant_quarter -> keep round(n/4) real objects
    variant_eighth  -> keep round(n/8) real objects
  (always keeps at least one object)

  Layout-scramble variant (seeded, deterministic):
    variant_scramble -> relocate every real object to a random position within
                        the room footprint, preserving the object set and each
                        object's rotation, destroying the arrangement.

  Worst-match variants (structured around a retrieval hook):
    variant_alt_0   -> swap each asset for the rank-0 (worst-ranked) match
    variant_alt_2   -> swap each asset for the rank-2 match (from worst end)
    variant_alt_4   -> swap each asset for the rank-4 match (from worst end)

  Substitution variants (structured around the same retrieval hook):
    variant_subst_within -> swap each asset for a different instance of the
                            SAME object category
    variant_subst_cross  -> swap each asset for an instance of a random
                            DIFFERENT object category

  The worst-match / substitution generators lazy-import a retrieval hook (which
  itself lazy-imports torch / CLIP). When retrieval is unavailable (no GPU, no
  downloaded assets/models), they degrade gracefully: they copy the scene
  unchanged and record the intended substitution under the `_vlmunr_alt_intent`
  / `_vlmunr_subst_intent` key so the rest of the pipeline and the tests still
  pass.

The room-prior filter, removal, scramble, and category logic are exposed as
pure, importable, unit-testable functions.
"""

import argparse
import copy
import json
import os
import random
import re
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

# I-Design room dimensions [length(x), width(y), height(z)] in meters. Matches
# the default used by test.py; override via generate_variants(room_dims=...).
DEFAULT_ROOM_DIMS = [4.0, 4.0, 2.5]

ALT_VARIANTS = {}  # VLMUNR: alt_* dropped

# Substitution modes routed through the same lazy retrieval hook.
SUBST_VARIANTS = {
    "variant_subst_within": "within",
    "variant_subst_cross": "cross",
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


def _object_volume(obj: dict) -> float:
    """Bounding-box volume (length * width * height) of an object, or 0.0 if
    its size is missing. Used to rank objects by footprint for the
    biggest-only variant."""
    size = obj.get("size_in_meters") or {}
    try:
        return float(size.get("length", 0.0)) * float(size.get("width", 0.0)) * float(size.get("height", 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_biggest_only_scene(real_objects: list) -> list:
    """Keep ONLY the single largest object (by bounding-box volume).

    Ties are broken by original order (the first maximal object wins). The
    returned list contains a deep copy of that one object. If there are no
    real objects, returns an empty list. This is the inverse of removal:
    instead of randomly dropping objects, it keeps the most spatially
    dominant one and discards the rest -- without regenerating the scene.
    """
    if not real_objects:
        return []
    best = max(real_objects, key=_object_volume)
    return [copy.deepcopy(best)]


def scramble_positions(real_objects: list, room_dims: list, seed: int) -> list:
    """Relocate every real object to a random position within the room footprint.

    Returns a NEW list of deep-copied objects (the input is not mutated). For
    each object, `position.x` is drawn uniformly from `[0, length]` and
    `position.y` from `[0, width]`, where `room_dims = [length, width, height]`
    (the I-Design convention: x=length, y=width, z=height). Each object's base
    height (`position.z`) and its rotation are left UNCHANGED, so only the
    arrangement is destroyed.

    Deterministic for a given `seed`: same object ids/count, positions within
    the room bounds, rotation preserved.
    """
    length, width = room_dims[0], room_dims[1]
    rng = random.Random(seed)
    out: list = []
    for obj in real_objects:
        new_obj = copy.deepcopy(obj)
        pos = new_obj.setdefault("position", {})
        pos["x"] = rng.uniform(0.0, length)
        pos["y"] = rng.uniform(0.0, width)
        # z (base height) and rotation are intentionally left untouched.
        out.append(new_obj)
    return out


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


def build_worst_object_scene(
    real_objects: list,
    base_scene_dir: str,
    variant_assets_dir: str,
    rank: int = 0,
) -> tuple[list, dict]:
    """Fork the scene and re-retrieve the WORST-CLIP-matching 3D asset for
    each object, without regenerating the scene graph.

    This 'hacks' the retrieval sorting algorithm in retrieve.py: the base
    scene picks the best CLIP match (rank 0 best-first); this variant instead
    picks the lowest-similarity candidate that still passes the asset filter
    (rank 0 from the worst end, or `rank` from the worst end). The scene
    graph (objects, positions, relations) is preserved unchanged; only the
    Assets are swapped.

    Each worst-match .glb is downloaded into `variant_assets_dir/<id>.glb`.
    If retrieval is unavailable (no GPU / no embeddings) or fails for an
    object, we fall back to copying the BASE scene's `<id>.glb` into the
    variant Assets dir so the variant still renders, and record the object
    under intent as "worst_fallback".

    Returns (scene_graph_list, intent_map).
    """
    import shutil as _sh

    os.makedirs(variant_assets_dir, exist_ok=True)
    out = [copy.deepcopy(o) for o in real_objects]
    intent: dict[str, str] = {}
    base_assets = os.path.join(os.path.abspath(base_scene_dir), "Assets")

    # Only attempt real retrieval when the heavy backend is ALREADY loaded.
    # Otherwise we would silently trigger a ~3GB embedding/CLIP download from
    # inside a variant generator. The CLI loads the backend explicitly before
    # calling this; on CPU-only hosts it stays None and every object falls back
    # to copying the base asset (the variant still renders).
    backend_ok = False
    try:
        import retrieve as _ret  # lazy

        backend_ok = _ret.backend_available()
    except Exception:
        backend_ok = False

    for obj in out:
        obj_id = obj["new_object_id"]
        retrieved_uid = None
        if backend_ok:
            try:
                # Use the object directly (it carries style/material for the
                # CLIP query). rank selects how far from the worst end.
                retrieved_uid = _ret.retrieve_asset_for_object(
                    obj, variant_assets_dir, match="worst", sim_th=0.0,
                )
            except Exception:  # backend unavailable for this object
                retrieved_uid = None

        glb_path = os.path.join(variant_assets_dir, f"{obj_id}.glb")
        if retrieved_uid is not None and os.path.exists(glb_path):
            obj["_vlmunr_worst_asset"] = retrieved_uid
            intent[obj_id] = "worst"
        else:
            # Fallback: reuse the base scene's asset so the variant still
            # renders. This keeps the variant runnable when retrieval is
            # unavailable (the common case on CPU-only hosts).
            base_glb = os.path.join(base_assets, f"{obj_id}.glb")
            if os.path.exists(base_glb):
                _sh.copy2(base_glb, glb_path)
                intent[obj_id] = "worst_fallback"
            else:
                intent[obj_id] = "worst_missing"
    return out, intent


def object_category(object_id: str) -> str:
    """Derive an object's category from its id.

    I-Design ids look like `chair_1`, `table_2`, `floor_lamp_3`. The category is
    the id with any trailing `_<digits>` instance suffix stripped (mirroring the
    digit-removal preprocessing in `retrieve.py`). Ids without a numeric suffix
    are returned unchanged.
    """
    return re.sub(r"_\d+$", "", object_id)


def categories_in_scene(real_objects: list) -> list:
    """Ordered, de-duplicated list of categories present in the scene."""
    seen: dict[str, None] = {}
    for obj in real_objects:
        seen.setdefault(object_category(obj["new_object_id"]), None)
    return list(seen)


def _try_retrieval_subst(
    object_id: str, mode: str, target_category: Optional[str], scene_dir: str
) -> Optional[str]:
    """Lazy-imported retrieval hook returning a substitute asset uid for an
    object, or None if retrieval is unavailable.

    `mode` is "within" (a different instance of the same category) or "cross"
    (an instance of `target_category`, a different category). Like
    `_try_retrieval_alt`, this imports heavy dependencies only when called and
    returns None on any failure so the caller can degrade gracefully. It is NOT
    exercised in unit tests / smoke runs (no GPU, no downloads).
    """
    try:  # pragma: no cover - requires GPU + model downloads
        import torch  # noqa: F401  (lazy, import-light at module level)
        from vlmunr_retrieval_hook import retrieve_substitute  # type: ignore

        return retrieve_substitute(object_id, mode, target_category, scene_dir)
    except Exception:
        return None


def build_subst_scene(
    real_objects: list, mode: str, scene_dir: str, seed: int = 42
) -> tuple[list, dict]:
    """Build a within-/cross-category substitution variant scene_graph.

    `mode` is "within" (swap each object's asset for a different instance of the
    SAME category) or "cross" (swap for a random DIFFERENT category). Each
    object's asset is swapped via the lazy retrieval hook when available;
    otherwise the scene is copied unchanged and the intent is recorded per
    object id as `{object_id: mode}` (cross-category also records the chosen
    target category as `{object_id: "cross:<category>"}`).

    Returns (scene_graph_list, intent_map). Deterministic for a given `seed`
    (the cross-category target choice is seeded).
    """
    if mode not in ("within", "cross"):
        raise ValueError(f"Unknown substitution mode: {mode!r}")

    out = [copy.deepcopy(o) for o in real_objects]
    all_categories = categories_in_scene(real_objects)
    rng = random.Random(seed)
    intent: dict[str, str] = {}
    for obj in out:
        obj_id = obj["new_object_id"]
        own_cat = object_category(obj_id)
        target_category: Optional[str] = None
        if mode == "cross":
            others = [c for c in all_categories if c != own_cat]
            target_category = rng.choice(others) if others else own_cat
        subst_uid = _try_retrieval_subst(obj_id, mode, target_category, scene_dir)
        if subst_uid is not None:
            obj["_vlmunr_subst_asset"] = subst_uid
        elif mode == "cross":
            intent[obj_id] = f"cross:{target_category}"
        else:
            intent[obj_id] = mode
    return out, intent


def _write_scene(out_dir: str, scene_graph: list, assets_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scene_graph.json"), "w") as f:
        json.dump(scene_graph, f, indent=2)
    # Record where the renderer should read Assets from (the original scene).
    with open(os.path.join(out_dir, "vlmunr_assets_dir.txt"), "w") as f:
        f.write(assets_dir)


def generate_variants(
    scene_dir: str, seed: int = 42, room_dims: Optional[list] = None
) -> dict:
    """Generate all variant directories as siblings of `scene_dir`.

    Returns a mapping {variant_name: variant_dir}. `room_dims` is the
    [length, width, height] footprint used by the layout-scramble variant and
    defaults to `DEFAULT_ROOM_DIMS`.
    """
    if room_dims is None:
        room_dims = DEFAULT_ROOM_DIMS
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

    # Layout-scramble variant (positions randomized, asset set preserved).
    scramble_scene = scramble_positions(real_objects, room_dims, seed)
    scramble_dir = os.path.join(parent, f"{base_name}_variant_scramble")
    _write_scene(scramble_dir, scramble_scene, assets_dir)
    created["variant_scramble"] = scramble_dir

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

    for name, mode in SUBST_VARIANTS.items():
        variant_scene, intent = build_subst_scene(
            real_objects, mode, scene_dir, seed
        )
        out_dir = os.path.join(parent, f"{base_name}_{name}")
        payload = list(variant_scene)
        if intent:
            payload = [{"_vlmunr_subst_intent": intent}] + payload
        _write_scene(out_dir, payload, assets_dir)
        created[name] = out_dir

    return created


# ----------------------------------------------------------------------------
# Named variants (CLI spec): variant_01_half, variant_02_biggest-only,
# variant_03_scrambled, variant_04_worst-object.
# ----------------------------------------------------------------------------

NAMED_VARIANTS = [
    "variant_01_half",
    "variant_02_biggest-only",
    "variant_03_scrambled",
    "variant_04_worst-object",
]


def _write_named_scene(out_dir: str, scene_graph: list, assets_dir: str, intent=None) -> None:
    """Write a named variant's scene_graph.json (a leading intent metadata
    entry is embedded when present so the file stays one JSON document)."""
    os.makedirs(out_dir, exist_ok=True)
    payload: list = list(scene_graph)
    if intent:
        payload = [{"_vlmunr_variant_intent": intent}] + payload
    with open(os.path.join(out_dir, "scene_graph.json"), "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "vlmunr_assets_dir.txt"), "w") as f:
        f.write(assets_dir)


def generate_named_variants(
    scene_dir: str,
    seed: int = 42,
    room_dims: Optional[list] = None,
    worst_rank: int = 0,
    variant_prefix: Optional[str] = None,
) -> dict:
    """Generate the four CLI-named variants as sibling dirs of `scene_dir`.

    Each variant forks the BASE scene (no regeneration):
      variant_01_half         -> keep round(n/2) real objects (seeded)
      variant_02_biggest-only -> keep the single largest object (by volume)
      variant_03_scrambled    -> randomize x/y of every object within the room
      variant_04_worst-object -> fork + re-retrieve worst-CLIP asset per object

    variant_04 writes its own Assets/ dir (worst-match .glb per object,
    falling back to copying the base asset when retrieval is unavailable).
    The other three share the base scene's Assets/ via vlmunr_assets_dir.txt.

    `variant_prefix` controls the directory name prefix. It defaults to
    `<basename of scene_dir>_` (the standalone-tool layout: sibling dirs
    named `<base>_variant_01_half`). Pass `variant_prefix=""` for the CLI
    run layout, where variants live INSIDE the run dir and are named simply
    `variant_01_half` etc.

    Returns {variant_name: variant_dir}.
    """
    if room_dims is None:
        room_dims = DEFAULT_ROOM_DIMS
    scene_path = os.path.join(scene_dir, "scene_graph.json")
    with open(scene_path) as f:
        scene_graph = json.load(f)
    real_objects = filter_real_objects(scene_graph)

    parent = os.path.dirname(os.path.abspath(scene_dir.rstrip(os.sep)))
    base_name = os.path.basename(os.path.abspath(scene_dir.rstrip(os.sep)))
    base_assets = os.path.join(os.path.abspath(scene_dir), "Assets")
    if variant_prefix is None:
        variant_prefix = f"{base_name}_"
    created: dict[str, str] = {}

    # variant_01_half
    half_scene = build_removal_scene(real_objects, 2, seed)
    out_dir = os.path.join(parent, f"{variant_prefix}variant_01_half")
    _write_named_scene(out_dir, half_scene, base_assets)
    created["variant_01_half"] = out_dir

    # variant_02_biggest-only
    biggest_scene = build_biggest_only_scene(real_objects)
    out_dir = os.path.join(parent, f"{variant_prefix}variant_02_biggest-only")
    _write_named_scene(out_dir, biggest_scene, base_assets)
    created["variant_02_biggest-only"] = out_dir

    # variant_03_scrambled
    scramble_scene = scramble_positions(real_objects, room_dims, seed)
    out_dir = os.path.join(parent, f"{variant_prefix}variant_03_scrambled")
    _write_named_scene(out_dir, scramble_scene, base_assets)
    created["variant_03_scrambled"] = out_dir

    # variant_04_worst-object (own Assets dir; re-retrieves worst-match assets)
    out_dir = os.path.join(parent, f"{variant_prefix}variant_04_worst-object")
    variant_assets = os.path.join(out_dir, "Assets")
    worst_scene, intent = build_worst_object_scene(
        real_objects, scene_dir, variant_assets, rank=worst_rank
    )
    _write_named_scene(out_dir, worst_scene, variant_assets, intent=intent)
    created["variant_04_worst-object"] = out_dir

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
