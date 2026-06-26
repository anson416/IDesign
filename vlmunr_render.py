"""Headless Blender renderer for the VLM-unreliability audit.

Loads an I-Design scene (scene_graph.json + Assets/<id>.glb), builds it in
Blender via the tested `vlmunr_bpa` renderer using the EXACT I-Design object
transform, then sweeps audit factors and writes PNGs to
`<scene-dir>/renderings/`.

Two-phase rendering per (res, focal, pitch, yaw, hdri):
  1. Render a transparent master `render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png`.
  2. Composite each background gray onto the master via
     `Renderer.add_bg_to_rgba` -> `render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png`.

The filename builders and scene-graph object iteration are pure functions so
they can be unit-tested without bpy.
"""

import argparse
import json
import math
import os
from typing import Optional

import vlmunr_config as cfg

# ----------------------------------------------------------------------------
# Pure helpers (no bpy) -- unit-testable.
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


def iter_real_objects(scene_graph: list):
    """Yield real placeable objects from a flat scene_graph list, skipping any
    metadata entries (e.g. _vlmunr_alt_intent) and room priors."""
    for item in scene_graph:
        if is_real_object(item):
            yield item


def master_filename(res: int, focal: int, pitch: int, yaw: int, hdri: str) -> str:
    """Filename of the transparent master render for a config."""
    return f"render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png"


def composite_filename(
    res: int, focal: int, bg: tuple, pitch: int, yaw: int, hdri: str
) -> str:
    """Filename of a background-composited render for a config."""
    r, g, b = bg
    return f"render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png"


def idesign_object_transform(item: dict) -> dict:
    """Compute the I-Design Blender transform parameters for an object.

    Replicates place_in_blender.py EXACTLY:
      location = (position.x, position.y, position.z)
      rotation about world Z by (z_angle / 180) * pi + pi  [radians]
      rescale so dims == (length, width, height)

    Returns a dict with:
      location: (x, y, z)
      rotation_z_rad: float (radians, includes the +pi/+180deg offset)
      rotation_z_deg: float (degrees, for bpa.transform which takes degrees)
      dims: (length, width, height)
    """
    pos = item["position"]
    location = (pos["x"], pos["y"], pos["z"])
    z_angle = item["rotation"]["z_angle"]
    rot_rad = (z_angle / 180.0) * math.pi + math.pi
    size = item["size_in_meters"]
    dims = (size["length"], size["width"], size["height"])
    return {
        "location": location,
        "rotation_z_rad": rot_rad,
        "rotation_z_deg": math.degrees(rot_rad),
        "dims": dims,
    }


# ----------------------------------------------------------------------------
# Blender scene building (requires bpy via vlmunr_bpa).
# ----------------------------------------------------------------------------


def _resolve_assets_dir(scene_dir: str, assets_dir: Optional[str]) -> str:
    if assets_dir:
        return assets_dir
    # Variant dirs record the original Assets path.
    marker = os.path.join(scene_dir, "vlmunr_assets_dir.txt")
    if os.path.exists(marker):
        with open(marker) as f:
            return f.read().strip()
    return os.path.join(scene_dir, "Assets")


def load_scene_into_blender(
    scene_dir: str, assets_dir: Optional[str] = None
) -> int:
    """Load scene_graph.json into the current Blender scene using vlmunr_bpa
    and the exact I-Design transform.

    Returns the number of objects placed.
    """
    import vlmunr_bpa as bpa  # lazy: needs bpy

    assets = _resolve_assets_dir(scene_dir, assets_dir)
    with open(os.path.join(scene_dir, "scene_graph.json")) as f:
        scene_graph = json.load(f)

    placed = 0
    for item in iter_real_objects(scene_graph):
        obj_id = item["new_object_id"]
        glb_path = os.path.join(assets, f"{obj_id}.glb")
        if not os.path.exists(glb_path):
            continue
        obj = _import_and_join_glb(bpa, glb_path, obj_id)
        if obj is None:
            continue
        _apply_idesign_transform(bpa, obj, item)
        placed += 1
    return placed


def _import_and_join_glb(bpa, glb_path: str, obj_id: str):
    """Import a .glb, join child meshes into one, set origin to BOUNDS.

    Returns the joined mesh Object, or None if no mesh was produced.
    """
    import bpy  # type: ignore

    before = set(bpy.context.scene.objects)
    with bpa.redirect_stdout():
        bpy.ops.import_scene.gltf(filepath=glb_path)
    new_objs = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in new_objs if o.type == "MESH"]
    if not meshes:
        # Clean up any empties imported.
        for o in new_objs:
            bpy.data.objects.remove(o, do_unlink=True)
        return None

    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active

    # Drop any leftover empties from the import.
    for o in new_objs:
        if o.type == "EMPTY" and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)

    bpa.focus(joined)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    joined.name = obj_id
    return joined


def _apply_idesign_transform(bpa, obj, item: dict) -> None:
    """Apply location, world-Z rotation (with +180deg offset), and rescale to
    target dims -- replicating place_in_blender.py."""
    import bpy  # type: ignore

    t = idesign_object_transform(item)
    bpa.focus(obj)
    obj.location = t["location"]
    bpy.ops.transform.rotate(value=t["rotation_z_rad"], orient_axis="Z")
    # Rescale to exact target dimensions.
    length, width, height = t["dims"]
    d = obj.dimensions
    sx = length / d.x if d.x else 1.0
    sy = width / d.y if d.y else 1.0
    sz = height / d.z if d.z else 1.0
    obj.scale = (sx, sy, sz)


# ----------------------------------------------------------------------------
# Rendering driver.
# ----------------------------------------------------------------------------


def render_phases(
    scene_dir: str,
    phases: list,
    assets_dir: Optional[str] = None,
) -> list:
    """Build the scene and render all configs for the given phases.

    Returns the list of output PNG paths written.
    """
    import vlmunr_bpa as bpa  # lazy: needs bpy

    out_root = os.path.join(scene_dir, "renderings")
    os.makedirs(out_root, exist_ok=True)

    # Build the scene once.
    bpa.clear()
    n = load_scene_into_blender(scene_dir, assets_dir=assets_dir)
    if n == 0:
        raise RuntimeError(f"No objects placed from {scene_dir!r}")

    renderer = bpa.Renderer()

    # Collect unique configs across phases (dedup identical baseline points).
    configs = []
    seen = set()
    for phase in phases:
        for c in cfg.phase_levels(phase):
            key = (
                c["res"],
                c["focal"],
                tuple(c["bg"]),
                c["hdri"],
                c["pitch"],
                c["yaw"],
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(c)

    written = _render_configs(bpa, renderer, configs, out_root)
    return written


def _render_configs(bpa, renderer, configs, out_root) -> list:
    """Render a list of configs with two-phase (master + composite) output.

    Re-initializes Blender world whenever the HDRI changes.
    """
    written = []
    current_hdri = None

    # Group by (res, focal, pitch, yaw, hdri) -> set of bg colors.
    masters: dict = {}
    for c in configs:
        mkey = (c["res"], c["focal"], c["pitch"], c["yaw"], c["hdri"])
        masters.setdefault(mkey, set()).add(tuple(c["bg"]))

    for (res, focal, pitch, yaw, hdri), bgs in masters.items():
        if hdri != current_hdri:
            bpa.initialize(
                transparent=True,
                environment_map=(cfg.hdri_path(hdri), 1.0),
            )
            current_hdri = hdri

        master_path = os.path.join(
            out_root, master_filename(res, focal, pitch, yaw, hdri)
        )
        center, radius = renderer.compute_bounding_sphere()
        renderer.render_perspective(
            master_path,
            center,
            radius,
            rotation=(pitch, 0, yaw),
            resolution=res,
            focal_length=focal,
            background=None,
        )
        written.append(master_path)

        for bg in bgs:
            comp_path = os.path.join(
                out_root,
                composite_filename(res, focal, bg, pitch, yaw, hdri),
            )
            renderer.add_bg_to_rgba(master_path, comp_path, color=bg)
            written.append(comp_path)

    return written


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Render an I-Design scene across audit factors."
    )
    parser.add_argument(
        "--scene-dir",
        required=True,
        help="Directory containing scene_graph.json and Assets/",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Override directory for Assets/<id>.glb (defaults to scene-dir/Assets "
        "or the path recorded by vlmunr_variants).",
    )
    parser.add_argument(
        "--phase",
        choices=cfg.ALL_PHASES + ["all"],
        default="all",
    )
    args = parser.parse_args(argv)

    phases = cfg.ALL_PHASES if args.phase == "all" else [args.phase]
    written = render_phases(args.scene_dir, phases, assets_dir=args.assets_dir)
    print(f"Wrote {len(written)} PNG(s) to {os.path.join(args.scene_dir, 'renderings')}")


if __name__ == "__main__":
    main()
