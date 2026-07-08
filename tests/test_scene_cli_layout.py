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

import pytest


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


# ---------------------------------------------------------------------------
# --render / --render-all / --path (render behavior, with render mocked).
# ---------------------------------------------------------------------------


def _mock_render(monkeypatch):
    """Replace vlmunr_render.render_scene with a recorder.

    Returns the list of (scene_dir, n_configs) calls it observed.
    """
    import sys
    scene_cli = importlib.import_module("scene_cli")
    # scene_cli imports vlmunr_render lazily inside _render_run, so patch the
    # real module object it imports.
    vrender = importlib.import_module("vlmunr_render")
    calls = []
    def _fake(scene_dir, configs, room_dims, assets_dir=None):
        # Touch a renderings/ dir so layout assertions can find it.
        os.makedirs(os.path.join(scene_dir, "renderings"), exist_ok=True)
        calls.append((scene_dir, len(configs), list(room_dims)))
        return []
    monkeypatch.setattr(vrender, "render_scene", _fake)
    # _render_run does `import vlmunr_render as vrender` then calls
    # vrender.render_scene -- so patching the module attribute is enough.
    return calls


def test_cli_render_prompt_renders_base(tmp_path, monkeypatch):
    """--prompt --render generates the base scene then renders it (1 config)."""
    scene_cli = importlib.import_module("scene_cli")
    outputs_root = tmp_path / "outputs"
    monkeypatch.setattr(scene_cli, "_generate_scene", lambda args, llm_config: _FakeIDesign())
    monkeypatch.setattr(scene_cli, "_maybe_load_retrieval_backend", lambda args: False)
    calls = _mock_render(monkeypatch)

    rc = scene_cli.main([
        "--prompt", "A cozy reading nook",
        "--api-key", "sk-test",
        "--render",
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    # Exactly one render call (base only; no --variants).
    assert len(calls) == 1
    scene_dir, n_configs, room_dims = calls[0]
    assert scene_dir.endswith("base")
    assert n_configs == 1  # --render == single baseline config
    assert room_dims == [4.0, 4.0, 2.5]
    # renderings/ folder created under base/.
    run_dir = outputs_root / os.listdir(outputs_root)[0]
    assert (run_dir / "base" / "renderings").is_dir()
    # config.json records the render flag.
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["render"] is True and cfg["render_all"] is False


def test_cli_render_with_variants_renders_each(tmp_path, monkeypatch):
    """--prompt --variants --render renders base + all 4 variant dirs."""
    scene_cli = importlib.import_module("scene_cli")
    outputs_root = tmp_path / "outputs"
    monkeypatch.setattr(scene_cli, "_generate_scene", lambda args, llm_config: _FakeIDesign())
    monkeypatch.setattr(scene_cli, "_maybe_load_retrieval_backend", lambda args: False)
    calls = _mock_render(monkeypatch)

    rc = scene_cli.main([
        "--prompt", "A cozy reading nook",
        "--api-key", "sk-test",
        "--variants", "--render",
        "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    # 1 base + 4 variants = 5 render calls.
    assert len(calls) == 5
    rendered_basenames = sorted(os.path.basename(c[0]) for c in calls)
    assert rendered_basenames == [
        "base", "variant_01_half", "variant_02_biggest-only",
        "variant_03_scrambled", "variant_04_worst-object",
    ]


def test_cli_render_all_uses_44_configs(tmp_path, monkeypatch):
    """--render-all passes the deduped 44-config sweep to render_scene."""
    import vlmunr_config as cfg
    scene_cli = importlib.import_module("scene_cli")
    outputs_root = tmp_path / "outputs"
    monkeypatch.setattr(scene_cli, "_generate_scene", lambda args, llm_config: _FakeIDesign())
    monkeypatch.setattr(scene_cli, "_maybe_load_retrieval_backend", lambda args: False)
    calls = _mock_render(monkeypatch)

    rc = scene_cli.main([
        "--prompt", "x", "--api-key", "sk-test",
        "--render-all", "--outputs-root", str(outputs_root),
    ])
    assert rc == 0
    assert calls[0][1] == len(cfg.all_render_configs())  # 44


def test_cli_path_mode_renders_existing_run_no_generation(tmp_path, monkeypatch):
    """--path <run> --render reads room_dims from config.json and renders base
    (+ variants present) WITHOUT running the LLM pipeline."""
    scene_cli = importlib.import_module("scene_cli")
    run_dir = tmp_path / "outputs" / "20260101-000000"
    (run_dir / "base").mkdir(parents=True)
    (run_dir / "base" / "scene_graph.json").write_text("[]")
    (run_dir / "base" / "Assets").mkdir()
    (run_dir / "variant_01_half").mkdir()
    (run_dir / "variant_01_half" / "scene_graph.json").write_text("[]")
    # config.json with a non-default room_dimensions.
    cfg_json = {
        "room_dimensions": [6.0, 5.0, 3.0], "render": True, "render_all": False,
        "prompt": "x", "llm": {"api_key": "***"},
    }
    (run_dir / "config.json").write_text(json.dumps(cfg_json))
    calls = _mock_render(monkeypatch)

    # If path mode tried to generate, _generate_scene (unpatched) would fail;
    # patch it to assert it is NEVER called.
    gen_called = []
    monkeypatch.setattr(scene_cli, "_generate_scene",
                        lambda *a, **k: gen_called.append(1) or (_ for _ in ()).throw(AssertionError("must not generate")))

    rc = scene_cli.main([
        "--path", str(run_dir), "--render",
    ])
    assert rc == 0
    assert gen_called == []  # no generation in path mode
    # base + variant_01_half rendered; the other 3 variant dirs are absent.
    rendered = sorted(os.path.basename(c[0]) for c in calls)
    assert rendered == ["base", "variant_01_half"]
    # room_dims read from config.json (6,5,3), not the default.
    assert calls[0][2] == [6.0, 5.0, 3.0]
    assert calls[0][1] == 1  # --render == single config


def test_cli_render_and_render_all_mutually_exclusive(tmp_path, monkeypatch):
    scene_cli = importlib.import_module("scene_cli")
    _mock_render(monkeypatch)
    with pytest.raises(SystemExit):
        scene_cli.main(["--prompt", "x", "--api-key", "sk-test",
                        "--render", "--render-all",
                        "--outputs-root", str(tmp_path / "o")])


def test_cli_prompt_and_path_mutually_exclusive(tmp_path):
    scene_cli = importlib.import_module("scene_cli")
    with pytest.raises(SystemExit):
        scene_cli.main(["--prompt", "x", "--path", str(tmp_path), "--render"])


def test_cli_path_requires_render_flag(tmp_path):
    scene_cli = importlib.import_module("scene_cli")
    with pytest.raises(SystemExit):
        scene_cli.main(["--path", str(tmp_path)])
