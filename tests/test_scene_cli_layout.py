"""Layout regression test for scene_cli.py.

Asserts the output directory tree matches the documented contract:

  outputs/<YYYYMMDD-HHMMSS>/
    config.json
    base/
      scene_graph.json
    variant_01_half/
    variant_02_biggest-only/
    variant_03_scrambled/
    variant_04_worst-object/

The LLM pipeline + retrieval are mocked out (no API calls, no downloads).
"""

import importlib
import json
import os
from unittest import mock


def _fake_real_object(oid, l=1.0, w=1.0, h=1.0):
    return {
        "new_object_id": oid,
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "rotation": {"z_angle": 0.0},
        "size_in_meters": {"length": l, "width": w, "height": h},
    }


def _fake_scene_graph():
    # 4 real objects + room priors (so variant_01_half keeps round(4/2)=2).
    return [
        {"new_object_id": "south_wall", "itemType": "wall"},
        {"new_object_id": "ceiling", "itemType": "ceiling"},
        _fake_real_object("obj_0", l=1.0, w=1.0, h=1.0),
        _fake_real_object("obj_1", l=2.0, w=2.0, h=2.0),
        _fake_real_object("obj_2", l=3.0, w=1.0, h=1.0),
        _fake_real_object("obj_3", l=1.0, w=4.0, h=1.0),
    ]


class _FakeIDesign:
    def __init__(self, *a, **kw):
        self.scene_graph = _fake_scene_graph()


def test_cli_produces_nested_layout_with_base_and_variants(tmp_path, monkeypatch):
    scene_cli = importlib.import_module("scene_cli")

    outputs_root = tmp_path / "outputs"

    # Mock the heavy/external pieces: LLM pipeline + retrieval backend.
    monkeypatch.setattr(scene_cli, "_generate_scene", lambda args, llm_config: _FakeIDesign())
    monkeypatch.setattr(scene_cli, "_maybe_load_retrieval_backend", lambda args: False)

    argv = [
        "--prompt", "A cozy reading nook",
        "--api-key", "sk-test",
        "--variants",
        "--outputs-root", str(outputs_root),
    ]
    rc = scene_cli.main(argv)
    assert rc == 0

    # Exactly one run dir under outputs/.
    runs = [p for p in os.listdir(outputs_root) if (outputs_root / p).is_dir()]
    assert len(runs) == 1, runs
    run_dir = outputs_root / runs[0]

    # config.json at the run root.
    assert (run_dir / "config.json").is_file()

    # base/ holds the scene graph (NOT at the run root).
    assert (run_dir / "base" / "scene_graph.json").is_file()
    assert not (run_dir / "scene_graph.json").exists(), "base scene must live under base/"

    # config records the prompt + masked key.
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["prompt"] == "A cozy reading nook"
    assert cfg["llm"]["api_key"] == "***"

    # The four named variants are direct children of the run dir (no run-id
    # prefix) and each carries its own scene_graph.json.
    for name in (
        "variant_01_half",
        "variant_02_biggest-only",
        "variant_03_scrambled",
        "variant_04_worst-object",
    ):
        vdir = run_dir / name
        assert vdir.is_dir(), name
        assert (vdir / "scene_graph.json").is_file(), name

    # variant_01_half keeps round(4/2)=2 real objects.
    half = json.loads((run_dir / "variant_01_half" / "scene_graph.json").read_text())
    import vlmunr_variants as variants
    assert len(variants.filter_real_objects(half)) == 2

    # variant_02_biggest-only keeps the single largest (obj_1, volume 8).
    big = json.loads((run_dir / "variant_02_biggest-only" / "scene_graph.json").read_text())
    reals = variants.filter_real_objects(big)
    assert len(reals) == 1 and reals[0]["new_object_id"] == "obj_1"

    # variant_04 has its own Assets/ dir.
    assert (run_dir / "variant_04_worst-object" / "Assets").is_dir()


def test_cli_without_variants_only_writes_base(tmp_path, monkeypatch):
    scene_cli = importlib.import_module("scene_cli")
    outputs_root = tmp_path / "outputs"
    monkeypatch.setattr(scene_cli, "_generate_scene", lambda args, llm_config: _FakeIDesign())
    monkeypatch.setattr(scene_cli, "_maybe_load_retrieval_backend", lambda args: False)

    rc = scene_cli.main([
        "--prompt", "A sparse study",
        "--api-key", "sk-test",
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    runs = [p for p in os.listdir(outputs_root) if (outputs_root / p).is_dir()]
    run_dir = outputs_root / runs[0]
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "base" / "scene_graph.json").is_file()
    # No variant dirs without --variants.
    assert not (run_dir / "variant_01_half").exists()
