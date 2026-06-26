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
        "variant_scramble",
        "variant_alt_0",
        "variant_alt_2",
        "variant_alt_4",
        "variant_subst_within",
        "variant_subst_cross",
    }
    for path in created.values():
        assert os.path.exists(os.path.join(path, "scene_graph.json"))

    # alt variant retains intent metadata but renderer ignores it.
    with open(os.path.join(created["variant_alt_0"], "scene_graph.json")) as f:
        alt = json.load(f)
    assert any("_vlmunr_alt_intent" in e for e in alt)
    assert len(list(render.iter_real_objects(alt))) == 16

    # subst variant retains its own intent metadata; renderer ignores it.
    with open(
        os.path.join(created["variant_subst_within"], "scene_graph.json")
    ) as f:
        subst = json.load(f)
    assert any("_vlmunr_subst_intent" in e for e in subst)
    assert len(list(render.iter_real_objects(subst))) == 16

    # scramble variant keeps the full object set, no metadata entry.
    with open(os.path.join(created["variant_scramble"], "scene_graph.json")) as f:
        scram = json.load(f)
    assert len(scram) == 16


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


def test_factor_level_counts_match_paper_table1():
    assert len(cfg.RESOLUTIONS) == 9
    assert len(cfg.FOCAL_LENGTHS) == 7
    assert len(cfg.PITCHES) == 7
    assert len(cfg.YAWS) == 8
    assert len(cfg.BACKGROUND_GRAYS) == 6
    assert len(cfg.BACKGROUND_CHROMATIC) == 3


def test_factor_level_values_match_paper_table1():
    assert cfg.RESOLUTIONS == [196, 224, 256, 336, 384, 448, 512, 768, 1024]
    assert cfg.FOCAL_LENGTHS == [16, 24, 35, 50, 85, 100, 200]
    assert cfg.BACKGROUND_GRAYS == [0, 65, 128, 186, 204, 255]
    assert cfg.BACKGROUND_CHROMATIC == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert cfg.PITCHES == [0, 15, 30, 45, 60, 75, 90]
    assert cfg.YAWS == [0, 45, 90, 135, 180, 225, 270, 315]
    # Floor-texture background is a documented, NON-rendered sentinel.
    assert cfg.FLOOR_TEXTURE_BACKGROUND == "floor_texture"
    assert cfg.BASELINE_YAW_PITCH == 45


def test_all_phases_contains_new_phases():
    assert cfg.ALL_PHASES == [
        "1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"
    ]


def test_phase_levels_counts():
    assert len(cfg.phase_levels("1a")) == len(cfg.RESOLUTIONS)
    assert len(cfg.phase_levels("1b")) == len(cfg.BACKGROUND_GRAYS)
    assert len(cfg.phase_levels("1c")) == len(cfg.HDRIS)
    assert len(cfg.phase_levels("1d")) == len(cfg.FOCAL_LENGTHS)
    assert len(cfg.phase_levels("2")) == len(cfg.PITCHES) * len(cfg.YAWS)


def test_phase_levels_chroma():
    chroma = cfg.phase_levels("1b_chroma")
    assert len(chroma) == 3
    assert [c["bg"] for c in chroma] == cfg.BACKGROUND_CHROMATIC
    # All other factors held at baseline.
    for c in chroma:
        assert c["res"] == cfg.BASELINE_RES
        assert c["focal"] == cfg.BASELINE_FOCAL
        assert c["hdri"] == cfg.BASELINE_HDRI
        assert c["pitch"] == cfg.BASELINE_PITCH
        assert c["yaw"] == cfg.BASELINE_YAW


def test_phase_levels_pitch_sweep():
    pitch = cfg.phase_levels("2_pitch")
    assert len(pitch) == 7
    assert [c["pitch"] for c in pitch] == cfg.PITCHES
    assert all(c["yaw"] == 0 for c in pitch)


def test_phase_levels_yaw_sweep_at_pitch_45():
    yaw = cfg.phase_levels("2_yaw")
    assert len(yaw) == 8
    assert all(c["pitch"] == 45 for c in yaw)
    assert [c["yaw"] for c in yaw] == cfg.YAWS


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
# Layout scramble.
# ---------------------------------------------------------------------------

ROOM_DIMS = [4.0, 4.0, 2.5]


def test_scramble_deterministic_with_seed():
    real = variants.filter_real_objects(synthetic_scene(8))
    a = variants.scramble_positions(real, ROOM_DIMS, seed=11)
    b = variants.scramble_positions(real, ROOM_DIMS, seed=11)
    c = variants.scramble_positions(real, ROOM_DIMS, seed=12)
    a_pos = [(o["position"]["x"], o["position"]["y"]) for o in a]
    b_pos = [(o["position"]["x"], o["position"]["y"]) for o in b]
    c_pos = [(o["position"]["x"], o["position"]["y"]) for o in c]
    assert a_pos == b_pos
    assert a_pos != c_pos


def test_scramble_positions_within_bounds():
    real = variants.filter_real_objects(synthetic_scene(20))
    out = variants.scramble_positions(real, ROOM_DIMS, seed=3)
    for o in out:
        assert 0.0 <= o["position"]["x"] <= ROOM_DIMS[0]
        assert 0.0 <= o["position"]["y"] <= ROOM_DIMS[1]


def test_scramble_preserves_ids_count_and_rotation():
    real = variants.filter_real_objects(synthetic_scene(10))
    # Give objects distinct rotations and base heights to assert preservation.
    for i, o in enumerate(real):
        o["rotation"]["z_angle"] = float(i * 9)
        o["position"]["z"] = float(i) * 0.1
    out = variants.scramble_positions(real, ROOM_DIMS, seed=5)
    assert len(out) == len(real)
    assert [o["new_object_id"] for o in out] == [o["new_object_id"] for o in real]
    for orig, new in zip(real, out):
        assert new["rotation"] == orig["rotation"]
        assert new["position"]["z"] == orig["position"]["z"]


def test_scramble_does_not_mutate_input():
    real = variants.filter_real_objects(synthetic_scene(4))
    before = [(o["position"]["x"], o["position"]["y"]) for o in real]
    variants.scramble_positions(real, ROOM_DIMS, seed=1)
    after = [(o["position"]["x"], o["position"]["y"]) for o in real]
    assert before == after


# ---------------------------------------------------------------------------
# Within-/cross-category substitution.
# ---------------------------------------------------------------------------


def category_scene():
    return [
        make_object("chair_1"),
        make_object("chair_2"),
        make_object("table_1"),
        make_object("floor_lamp_1"),
    ]


def test_object_category_strips_instance_suffix():
    assert variants.object_category("chair_1") == "chair"
    assert variants.object_category("floor_lamp_3") == "floor_lamp"
    assert variants.object_category("rug") == "rug"


def test_categories_in_scene_dedup_ordered():
    real = category_scene()
    assert variants.categories_in_scene(real) == ["chair", "table", "floor_lamp"]


def test_subst_within_records_intent_on_degradation():
    real = category_scene()
    scene, intent = variants.build_subst_scene(
        real, "within", scene_dir="/nonexistent", seed=1
    )
    assert len(scene) == 4
    assert intent == {o["new_object_id"]: "within" for o in real}


def test_subst_cross_records_target_category_intent():
    real = category_scene()
    scene, intent = variants.build_subst_scene(
        real, "cross", scene_dir="/nonexistent", seed=1
    )
    assert len(scene) == 4
    # Every object records a cross-category target that differs from its own.
    for o in real:
        obj_id = o["new_object_id"]
        own_cat = variants.object_category(obj_id)
        recorded = intent[obj_id]
        assert recorded.startswith("cross:")
        target = recorded.split(":", 1)[1]
        assert target != own_cat
        assert target in variants.categories_in_scene(real)


def test_subst_cross_deterministic_with_seed():
    real = category_scene()
    _, a = variants.build_subst_scene(real, "cross", "/nonexistent", seed=7)
    _, b = variants.build_subst_scene(real, "cross", "/nonexistent", seed=7)
    _, c = variants.build_subst_scene(real, "cross", "/nonexistent", seed=8)
    assert a == b
    # Different seed should (with high probability) change at least one target.
    assert a != c


def test_subst_unknown_mode_raises():
    with pytest.raises(ValueError):
        variants.build_subst_scene(category_scene(), "sideways", "/nonexistent")


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
