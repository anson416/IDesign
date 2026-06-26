"""Integration tests for the VLM-unreliability audit harness.

Covers:
  (a) filename builders (exact strings)
  (b) room-prior filtering and removal fractions on a synthetic scene_graph
  (c) the I-Design transform math as pure functions
  (d) a bpy SMOKE test: build a synthetic primitive-cube scene and render one
      config, asserting a non-empty PNG is produced.
"""

import importlib
import math
import os

import pytest

import vlmunr_config as cfg
import vlmunr_render as render
import vlmunr_variants as variants


# ---------------------------------------------------------------------------
# Synthetic scene fixtures (in-memory).
# ---------------------------------------------------------------------------


def make_object(obj_id, x=0.0, y=0.0, z=0.0, z_angle=0.0, l=1.0, w=1.0, h=1.0):
    return {
        "new_object_id": obj_id,
        "position": {"x": x, "y": y, "z": z},
        "rotation": {"z_angle": z_angle},
        "size_in_meters": {"length": l, "width": w, "height": h},
    }


def make_room_prior(obj_id, item_type):
    return {"new_object_id": obj_id, "itemType": item_type}


def synthetic_scene(n_objects):
    sg = [
        make_room_prior("south_wall", "wall"),
        make_room_prior("north_wall", "wall"),
        make_room_prior("middle of the room", "floor"),
        make_room_prior("ceiling", "ceiling"),
    ]
    for i in range(n_objects):
        sg.append(make_object(f"obj_{i}", x=float(i)))
    return sg


# ---------------------------------------------------------------------------
# (a) Filename builders.
# ---------------------------------------------------------------------------


def test_master_filename_exact():
    assert (
        render.master_filename(512, 50, 0, 0, "city")
        == "render_512_50_0_0_city.png"
    )
    assert (
        render.master_filename(1024, 200, 90, 330, "sunset")
        == "render_1024_200_90_330_sunset.png"
    )


def test_composite_filename_exact():
    assert (
        render.composite_filename(512, 50, (128, 128, 128), 0, 0, "city")
        == "render_512_50_128_128_128_0_0_city.png"
    )
    assert (
        render.composite_filename(224, 24, (0, 18, 255), 60, 120, "forest")
        == "render_224_24_0_18_255_60_120_forest.png"
    )


# ---------------------------------------------------------------------------
# (b) Room-prior filtering + removal fractions.
# ---------------------------------------------------------------------------


def test_filter_real_objects_drops_room_priors():
    sg = synthetic_scene(5)
    real = variants.filter_real_objects(sg)
    assert len(real) == 5
    assert all(variants.is_real_object(o) for o in real)
    assert all("itemType" not in o for o in real)


def test_filter_drops_named_room_priors_without_itemtype():
    sg = [make_object("south_wall"), make_object("sofa")]
    # south_wall id is a room prior even if it lacks itemType
    real = variants.filter_real_objects(sg)
    assert [o["new_object_id"] for o in real] == ["sofa"]


def test_keep_count_fractions():
    assert variants.keep_count(16, 2) == 8
    assert variants.keep_count(16, 4) == 4
    assert variants.keep_count(16, 8) == 2
    # rounding
    assert variants.keep_count(15, 2) == 8  # round(7.5)=8
    assert variants.keep_count(15, 8) == 2  # round(1.875)=2
    # always >= 1
    assert variants.keep_count(3, 8) == 1
    assert variants.keep_count(1, 8) == 1
    assert variants.keep_count(0, 2) == 0


def test_removal_counts_match_fractions():
    real = variants.filter_real_objects(synthetic_scene(16))
    half = variants.build_removal_scene(real, 2, seed=42)
    quarter = variants.build_removal_scene(real, 4, seed=42)
    eighth = variants.build_removal_scene(real, 8, seed=42)
    assert len(half) == 8
    assert len(quarter) == 4
    assert len(eighth) == 2


def test_removal_deterministic_with_seed():
    real = variants.filter_real_objects(synthetic_scene(16))
    a = [o["new_object_id"] for o in variants.build_removal_scene(real, 4, seed=7)]
    b = [o["new_object_id"] for o in variants.build_removal_scene(real, 4, seed=7)]
    c = [o["new_object_id"] for o in variants.build_removal_scene(real, 4, seed=8)]
    assert a == b
    assert a != c  # different seed -> different (with high probability) subset


def test_removal_always_keeps_at_least_one():
    real = variants.filter_real_objects(synthetic_scene(3))
    eighth = variants.build_removal_scene(real, 8, seed=1)
    assert len(eighth) == 1


def test_build_alt_scene_degrades_gracefully():
    real = variants.filter_real_objects(synthetic_scene(4))
    scene, intent = variants.build_alt_scene(real, rank=2, scene_dir="/nonexistent")
    # Retrieval unavailable -> scene unchanged, intent recorded for each object.
    assert len(scene) == 4
    assert intent == {f"obj_{i}": 2 for i in range(4)}


def test_generate_variants_writes_six_dirs(tmp_path):
    scene_dir = tmp_path / "scene"
    (scene_dir / "Assets").mkdir(parents=True)
    import json

    with open(scene_dir / "scene_graph.json", "w") as f:
        json.dump(synthetic_scene(16), f)

    created = variants.generate_variants(str(scene_dir), seed=42)
    assert set(created) == {
        "variant_half",
        "variant_quarter",
        "variant_eighth",
        "variant_alt_0",
        "variant_alt_2",
        "variant_alt_4",
    }
    for path in created.values():
        assert os.path.exists(os.path.join(path, "scene_graph.json"))

    # alt variant retains intent metadata but renderer ignores it.
    with open(os.path.join(created["variant_alt_0"], "scene_graph.json")) as f:
        alt = json.load(f)
    assert any("_vlmunr_alt_intent" in e for e in alt)
    assert len(list(render.iter_real_objects(alt))) == 16


# ---------------------------------------------------------------------------
# (c) I-Design transform math.
# ---------------------------------------------------------------------------


def test_idesign_transform_location_and_dims():
    item = make_object("x", x=1.0, y=2.0, z=3.0, l=0.5, w=0.6, h=0.7)
    t = render.idesign_object_transform(item)
    assert t["location"] == (1.0, 2.0, 3.0)
    assert t["dims"] == (0.5, 0.6, 0.7)


def test_idesign_transform_rotation_offset():
    # z_angle=0 -> (0/180)*pi + pi == pi radians (180 deg offset).
    item = make_object("x", z_angle=0.0)
    t = render.idesign_object_transform(item)
    assert math.isclose(t["rotation_z_rad"], math.pi)
    assert math.isclose(t["rotation_z_deg"], 180.0)

    # z_angle=90 -> (90/180)*pi + pi = 1.5*pi.
    item90 = make_object("x", z_angle=90.0)
    t90 = render.idesign_object_transform(item90)
    assert math.isclose(t90["rotation_z_rad"], 1.5 * math.pi)
    assert math.isclose(t90["rotation_z_deg"], 270.0)


# ---------------------------------------------------------------------------
# config / phase_levels.
# ---------------------------------------------------------------------------


def test_phase_levels_counts():
    assert len(cfg.phase_levels("1a")) == len(cfg.RESOLUTIONS)
    assert len(cfg.phase_levels("1b")) == len(cfg.BACKGROUND_GRAYS)
    assert len(cfg.phase_levels("1c")) == len(cfg.HDRIS)
    assert len(cfg.phase_levels("1d")) == len(cfg.FOCAL_LENGTHS)
    assert len(cfg.phase_levels("2")) == len(cfg.PITCHES) * len(cfg.YAWS)


def test_phase_levels_hold_baseline():
    for c in cfg.phase_levels("1a"):
        assert c["focal"] == cfg.BASELINE_FOCAL
        assert c["bg"] == cfg.BASELINE_BG
        assert c["hdri"] == cfg.BASELINE_HDRI
        assert c["pitch"] == cfg.BASELINE_PITCH
        assert c["yaw"] == cfg.BASELINE_YAW
    # 1b sweeps bg as gray triples.
    grays = {c["bg"] for c in cfg.phase_levels("1b")}
    assert (0, 0, 0) in grays and (255, 255, 255) in grays


def test_phase_levels_unknown_raises():
    with pytest.raises(ValueError):
        cfg.phase_levels("9z")


# ---------------------------------------------------------------------------
# (d) bpy SMOKE test.
#
# vlmunr_bpa redirects the real stdout file descriptor during rendering, which
# is incompatible with pytest's in-process output capture. We therefore run the
# synthetic-cube smoke render in a clean subprocess (vlmunr_smoke.py) and assert
# on its output and the PNGs it writes.
# ---------------------------------------------------------------------------

bpy = pytest.importorskip("bpy")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_smoke_render_primitive_cube(tmp_path):
    """Render ONE config on a synthetic 2-cube scene (no real assets) via a
    subprocess, asserting a non-empty master + composite PNG are produced."""
    import subprocess
    import sys

    out_root = tmp_path / "renderings"
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "vlmunr_smoke.py"), str(out_root)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"smoke render failed:\n{proc.stdout}\n{proc.stderr}"
    assert "SMOKE_OK" in proc.stdout

    master = out_root / render.master_filename(224, 50, 0, 0, "city")
    comp = out_root / render.composite_filename(
        224, 50, (128, 128, 128), 0, 0, "city"
    )
    assert master.exists() and master.stat().st_size > 0
    assert comp.exists() and comp.stat().st_size > 0
