"""cli.py — I-Design scene generation from a textual description.

This is the front entry point for the I-Design method. It runs the full
multi-agent LLM pipeline (Interior Designer -> Interior Architect -> Engineer
-> Layout Corrector -> Layout Refiner -> backtracking solver) from a free-text
room prompt, writes the base scene, and optionally retrieves 3D assets and
produces the four content variants.

==============================================================================
EXTERNAL LOCAL RESOURCES THIS METHOD REQUIRES
==============================================================================

  ENVIRONMENT (Python 3.13 required — bpy 5.1.2 is cp313-only; the `idesign`
  conda env is py3.13 and holds bpy + AG2 + the retrieval deps together).
  Dependency specs live in `pyproject.toml`. From a fresh py3.13 env:

      pip install -e ".[all]"      # generate + retrieve + render + viz + test
      # or pick extras: [render] [retrieve] [viz] [dev]

  STAGE A — Scene-graph generation (the LLM pipeline)
    Needs ONLY an OpenAI-compatible chat API. No databases, no GPU, no
    downloads. This always runs when you invoke the CLI.

  STAGE B — 3D-asset retrieval (OpenShape + CLIP + Objaverse)
    Needs the resources below. This runs ONLY when you pass `--retrieve`. If
    you omit it, the base scene_graph.json + variants are still produced; you
    just get no .glb assets (so Blender rendering would have nothing to show).

The resources for STAGE B (and how to prepare them if you don't have them):

  1. OpenShape embedding bank  (~1.5 GB on disk)
     - Repo: https://huggingface.co/OpenShape/openshape-objaverse-embeddings
       (GATED — you must request access and supply a HuggingFace token.)
     - Files: objaverse_meta.json (~440 MB) + objaverse.pt (~1 GB). These map
       every Objaverse object's uid to a CLIP-aligned embedding vector. CLIP
       text queries are matched against this bank to pick the best/worst asset.
     - Location: defaults to ./OpenShape-Embeddings. Override with
       --embeddings-dir. Downloaded automatically on first `--retrieve` run
       IF you have a valid HF token (pass --hf-token or run `huggingface-cli
       login`, or export HF_TOKEN).
     - PREPARE (one time):
         pip install huggingface_hub
         huggingface-cli login            # paste a token with read access
         # (accept the model gating on the repo webpage first)
         huggingface-cli download \
             OpenShape/openshape-objaverse-embeddings \
             --repo-type dataset \
             --local-dir ./OpenShape-Embeddings
       Then point the CLI at it: --embeddings-dir ./OpenShape-Embeddings

  2. CLIP text encoder  (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k, ~3.5 GB)
     - Downloaded automatically by `transformers.from_pretrained` into the
       HuggingFace cache (~/.cache/huggingface) on first `--retrieve` run.
     - No gating. Needs network on first run; cached afterwards.
     - PREPARE: nothing manual — just be online the first time. To pre-fetch:
         python -c "from transformers import CLIPModel, CLIPProcessor; \
           CLIPModel.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'); \
           CLIPProcessor.from_pretrained('laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')"

  3. Objaverse asset cache  (the actual .glb 3D models; grows with use)
     - Each retrieved object's .glb is downloaded by objaverse.load_objects
       on demand (only the objects your scene references — usually <1 MB each).
     - Location: defaults to ~/.objaverse_cache. Override with
       --objaverse-cache-dir (useful if your home dir is quota-limited).
     - PREPARE: nothing manual — assets download lazily during `--retrieve`.
       If you want a pre-populated cache, run a `--retrieve` once; subsequent
       runs reuse cached .glb files.

  4. Python package `objaverse`  (pip install objaverse)
     - Already in the `idesign` conda env. Required for any asset download.

  5. (OPTIONAL) `openshape` point-cloud encoder
     - NOT used by the text->asset retrieval path (that uses only the CLIP
       text encoder + the precomputed bank). It is imported only when the env
       var VLMUNR_LOAD_PC_ENCODER=1 is set. You can ignore it for scene
       generation; omit that env var (the default).

  6. Blender / `bpy` 5.1.2 — required only for `--render` / `--render-all`.
     - Installed in the `idesign` env (Python 3.13; bpy 5.1.2 is cp313-only).
       `cli.py` imports bpy directly and renders in-process; no separate
       Blender-bundled Python is needed. Omit the render flags if you only
       want scene_graph.json + Assets/ + variants.
     - The `idesign` env also uses AG2 (`ag2`, alias of the `autogen` package)
       instead of the old pyautogen 0.2.x: AG2 supports Python 3.13 AND keeps
       the 0.2-era `import autogen` API, so generation works unchanged.

  7. LLM API credentials — required for STAGE A.
     - An OpenAI-compatible endpoint. Pass --api-key, --base-url, --model,
       --temperature (and --json-model for the JSON-mode agents). Or export
       OPENAI_API_KEY / OPENAI_BASE_URL / CHATANYWHERE_API_KEY and omit the
       flags. Defaults target the chatanywhere proxy (gpt-5.1).

==============================================================================
QUICK START
==============================================================================

  # Activate the env that has autogen + the retrieval deps:
  conda activate idesign

  # Minimal: scene graph only (no assets, no variants):
  python cli.py --prompt "A creative vibrant living room" \
      --api-key "$OPENAI_API_KEY" --base-url https://api.chatanywhere.tech/v1 \
      --model gpt-5.1-2025-11-13 --temperature 0.7

  # Full: scene + best-match assets + all four variants:
  python cli.py --prompt "A cozy reading nook" \
      --api-key "$OPENAI_API_KEY" --base-url https://api.chatanywhere.tech/v1 \
      --model gpt-5.1-2025-11-13 --temperature 0.7 \
      --no-of-objects 12 --room-dims 4.0 4.0 2.5 --seed 42 \
      --variants --retrieve \
      --embeddings-dir ./OpenShape-Embeddings \
      --objaverse-cache-dir ~/objaverse_cache \
      --hf-token "$HF_TOKEN"

  # Generate + render the base (single baseline config) in one go:
  python cli.py --prompt "A cozy reading nook" --api-key "$OPENAI_API_KEY" \
      --retrieve --variants --render

  # Generate + render the full 6-factor sweep (44 configs) on base + variants:
  python cli.py --prompt "A cozy reading nook" --api-key "$OPENAI_API_KEY" \
      --retrieve --variants --render-all

  # Render an ALREADY-generated run (no generation; renders base + variant_*):
  python cli.py --path outputs/20260708-023434 --render-all

RENDERING
---------
Rendering uses Blender (bpy 5.1.2) in-process — the `idesign` env is Python 3.13
and holds bpy + AG2 (autogen) + the retrieval deps together, so `cli.py` does
generation and rendering in one process.

  --render        render every scene at the SINGLE baseline config:
                  res=512, bg=(255,255,255), env=city, focal=50mm, pitch=0, yaw=0.
  --render-all    render every scene across the 6 factor sweeps (44 configs
                  after dedup): resolution, focal, pitch(yaw 0), yaw(pitch 45),
                  env map, background color.
  (--render and --render-all are mutually exclusive.)

Two-phase per (res,focal,pitch,yaw,env): a transparent master (env-map lit) is
rendered at fit_ratio=1 (tight-fit), then each background color is PIL-composited
over it. The architectural shell (floor + 4 walls from --room-dims) is kept and
back-face culled (dollhouse) so oblique views see into the room. Output goes to
<scene>/renderings/ (ALWAYS named "renderings"):

  render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{env}.png
  render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{env}_bg-{r}-{g}-{b}.png

`--path` (mutually exclusive with --prompt) targets an existing run dir and
requires --render or --render-all: it renders base + any variant_* dirs present
without running the LLM pipeline (room_dims read from <run>/config.json).

Output layout (under outputs/<YYYYMMDD-HHMMSS-UTC>/):
  config.json                      prompt + llm config (api key masked) + metadata
  base/
    scene_graph.json               BASE scene (flat list: real objects + room priors)
    Assets/                        best-match .glb per object (with --retrieve)
    renderings/                    (with --render/--render-all) ALWAYS named "renderings"
  variant_01_half/                 keep round(n/2) objects (seeded, no regen)
  variant_02_biggest-only/         keep the single largest object (by volume)
  variant_03_scrambled/            randomize positions within the room (no regen)
  variant_04_worst-object/         fork + worst-CLIP asset per object (own Assets/)

Variants are cheap JSON transforms / asset swaps of the already-generated
base scene — they do NOT re-run the LLM (no extra cost). variant_04 reuses the
base asset and records "worst_fallback" intent when retrieval is unavailable.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Optional


def _utc_run_id() -> str:
    """Folder name from the current UTC time, e.g. 20260708-023434.

    Uses timezone-aware UTC. (The spec wrote ``datetime.datetime.now(datetime.UTC)``
    which only resolves under ``from datetime import datetime``; ``timezone.utc``
    is the stable, version-independent spelling under ``import datetime``.)
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S")


def _build_llm_config(args):
    from agents import LLMConfig

    return LLMConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        json_model=args.json_model,
    )


def _generate_scene(args, llm_config):
    """Run the I-Design pipeline and return the populated IDesign object."""
    from IDesign import IDesign

    i_design = IDesign(
        no_of_objects=args.no_of_objects,
        user_input=args.prompt,
        room_dimensions=list(args.room_dims),
        llm_config=llm_config,
    )
    # Phase 1: Interior Designer, Interior Architect, Engineer.
    i_design.create_initial_design()
    # Phase 2: Layout Corrector (spatial + size conflict pruning).
    i_design.correct_design(verbose=args.verbose)
    # Phase 3: Layout Refiner (child-object relative placement).
    i_design.refine_design(verbose=args.verbose)
    # Phase 4: Backtracking solver -> absolute positions.
    i_design.create_object_clusters(verbose=args.verbose)
    i_design.backtrack(verbose=args.verbose)
    return i_design


def _write_config(out_dir, args, run_id, n_real, retrieval_used):
    """Persist the prompt + llm config + run metadata to config.json."""
    config = {
        "run_id": run_id,
        "prompt": args.prompt,
        "no_of_objects": args.no_of_objects,
        "room_dimensions": list(args.room_dims),
        "seed": args.seed,
        "variants": args.variants,
        "retrieve_assets": args.retrieve,
        "render": getattr(args, "render", False),
        "render_all": getattr(args, "render_all", False),
        "n_real_objects": n_real,
        "llm": {
            "model": args.model,
            "base_url": args.base_url,
            "api_key": "***" if args.api_key else "",
            "temperature": args.temperature,
            "json_model": args.json_model,
        },
        "retrieval_backend_loaded": retrieval_used,
        "created_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _maybe_load_retrieval_backend(args):
    """Load the OpenShape/CLIP backend iff --retrieve was passed. Returns
    True if the backend is available, else False (graceful degradation)."""
    if not args.retrieve:
        return False
    try:
        import retrieve

        # The OpenShape embedding repo is gated; honour an explicit token,
        # else rely on a cached HF login / HF_TOKEN env.
        hf_token = getattr(args, "hf_token", None)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        embeddings_dir = getattr(args, "embeddings_dir", None)
        objaverse_cache_dir = getattr(args, "objaverse_cache_dir", None)
        retrieve.load_backend(
            embeddings_dir=embeddings_dir,
            objaverse_cache_dir=objaverse_cache_dir,
        )
        return retrieve.backend_available()
    except Exception as e:
        print(
            f"[cli] retrieval backend unavailable ({e}); "
            "skipping asset download (scenes still written)."
        )
        return False


def _retrieve_base_assets(base_dir, scene_graph):
    """Download the best-match .glb for every real object into base/Assets/."""
    import retrieve

    assets_dir = os.path.join(base_dir, "Assets")
    result = retrieve.retrieve_scene_assets(
        scene_graph,
        assets_dir,
        match="best",
        sim_th=0.1,
        verbose=True,
        autoload=False,  # backend already loaded by _maybe_load_retrieval_backend
    )
    print(
        f"[cli] base assets: placed={len(result['placed'])} "
        f"skipped={len(result['skipped'])}"
    )
    return assets_dir


def _make_variants(run_dir, base_dir, args, room_dims):
    """Generate the four named variants as children of the run dir.

    Variants fork the base scene (under base/); they share the base Assets/
    via a recorded marker file, except variant_04 which writes its own Assets/.
    With variant_prefix="" the dirs are named simply variant_01_half etc.
    (no run-id prefix) so they sit directly under <run>/.
    """
    import vlmunr_variants as variants

    created = variants.generate_named_variants(
        base_dir,
        seed=args.seed,
        room_dims=room_dims,
        worst_rank=0,
        variant_prefix="",
    )
    for name, path in created.items():
        print(f"[cli] variant {name}: {path}")


def _render_configs_for(mode: str):
    """Return the list of render configs for `--render` (single) or
    `--render-all` (the 6 sweeps). Returns None for an unknown mode."""
    import vlmunr_config as cfg

    if mode == "render":
        return cfg.single_render_config()
    if mode == "render_all":
        return cfg.all_render_configs()
    return None


def _scene_dirs_to_render(run_dir: str, want_variants: bool) -> list:
    """The scene dirs to render under a run dir: always base/, plus each
    variant_*/ dir present (only when --variants was used / they exist)."""
    import vlmunr_config as cfg  # noqa: F401  (kept for NAMED_VARIANTS parity)

    dirs = []
    base_dir = os.path.join(run_dir, "base")
    if os.path.isdir(base_dir):
        dirs.append(base_dir)
    if want_variants:
        import vlmunr_variants as variants

        for name in variants.NAMED_VARIANTS:
            vdir = os.path.join(run_dir, name)
            if os.path.isdir(vdir):
                dirs.append(vdir)
    return dirs


def _resolve_scene_assets_dir(scene_dir: str) -> str:
    """Resolve where a scene's .glb assets live, mirroring the renderer's
    `_resolve_assets_dir` in vlmunr_render.py.

    A scene either has its own Assets/ dir (base, variant_04) or records the
    original Assets path in vlmunr_assets_dir.txt (variant_01-03, which share
    the base Assets/). Falls back to <scene_dir>/Assets when neither applies.
    """
    marker = os.path.join(scene_dir, "vlmunr_assets_dir.txt")
    if os.path.exists(marker):
        try:
            with open(marker) as f:
                resolved = f.read().strip()
            if resolved:
                return resolved
        except OSError:
            pass
    return os.path.join(scene_dir, "Assets")


def _render_run(
    run_dir: str, room_dims: list, mode: str, want_variants: bool
) -> None:
    """Render base (+ variants if present) of a run dir with the given mode."""
    import vlmunr_render as vrender

    configs = _render_configs_for(mode)
    if not configs:
        print(f"[cli] unknown render mode: {mode}", file=sys.stderr)
        return
    scene_dirs = _scene_dirs_to_render(run_dir, want_variants)
    for sd in scene_dirs:
        # Warn if a scene has no resolvable Assets (nothing to show but the
        # shell). Variant dirs 01-03 share the base Assets/ via a
        # vlmunr_assets_dir.txt marker instead of a local Assets/ dir, so we
        # must resolve through the marker -- exactly as the renderer does --
        # not just check <sd>/Assets.
        assets = _resolve_scene_assets_dir(sd)
        if not os.path.isdir(assets) or not any(
            f.endswith(".glb")
            for f in os.listdir(assets)
            if os.path.isfile(os.path.join(assets, f))
        ):
            print(
                f"[cli] WARNING: no .glb assets in {assets} — rendering "
                "the shell only (did you forget --retrieve?).",
                file=sys.stderr,
            )
        print(f"[cli] rendering {sd} ({mode}, {len(configs)} configs)...")
        written = vrender.render_scene(sd, configs, list(room_dims))
        print(f"[cli]   wrote {len(written)} PNG(s) to {sd}/renderings")


def _room_dims_from_config(run_dir: str) -> list:
    """Read room_dimensions from <run>/config.json (path mode), falling back to
    the I-Design default if absent."""
    import vlmunr_config as cfg

    cfg_path = os.path.join(run_dir, "config.json")
    try:
        with open(cfg_path) as f:
            c = json.load(f)
        rd = c.get("room_dimensions")
        if isinstance(rd, list) and len(rd) == 3:
            return [float(v) for v in rd]
    except Exception:
        pass
    return [4.0, 4.0, 2.5]


def main(argv: Optional[list] = None) -> int:
    # On a headless server there is no X display; matplotlib's default backend
    # may try to open a window. Force the non-interactive Agg backend (always
    # available, unlike cv2's "offscreen" Qt plugin) so plt calls never block
    # or crash. The viz helpers in utils.py additionally skip cv2.imshow on
    # headless and write PNGs instead.
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MPLBACKEND", "Agg")

    parser = argparse.ArgumentParser(
        description="Generate and/or render an I-Design scene."
    )
    # --prompt and --path are mutually exclusive; exactly one is required.
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prompt",
        default=None,
        help="Free-text scene/room description (generates a new scene).",
    )
    src.add_argument(
        "--path",
        default=None,
        help="An existing run dir (outputs/<datetime>/) to render without "
        "generating. Must be combined with --render or --render-all.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1-2025-11-13",
        help="LLM model name for the chat agents.",
    )
    parser.add_argument(
        "--json-model",
        default="gpt-4o",
        help="LLM model name for JSON-mode agent calls "
        "(engineer + schema debugger).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"
        ),
        help="LLM API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CHATANYWHERE_API_KEY", ""),
        help="LLM API key.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature applied to ALL agents.",
    )
    parser.add_argument(
        "--no-of-objects",
        type=int,
        default=15,
        help="Target number of objects in the scene.",
    )
    parser.add_argument(
        "--room-dims",
        nargs=3,
        type=float,
        default=[4.0, 4.0, 2.5],
        metavar=("LENGTH", "WIDTH", "HEIGHT"),
        help="Room dimensions in meters (x, y, z).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic variants.",
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Also write the four content variants "
        "(variant_01_half, variant_02_biggest-only, "
        "variant_03_scrambled, variant_04_worst-object).",
    )
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Download 3D assets (best match for base, worst "
        "match for variant_04) via OpenShape/CLIP. Needs "
        "GPU + ~3GB embeddings; degrades gracefully.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=None,
        help="Directory for the OpenShape embedding bank "
        "(objaverse_meta.json + objaverse.pt). Defaults to "
        "./OpenShape-Embeddings (downloaded on first use).",
    )
    parser.add_argument(
        "--objaverse-cache-dir",
        default=None,
        help="Directory for the objaverse .glb cache "
        "(downloaded assets). Defaults to ~/.objaverse_cache.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token for the (gated) OpenShape "
        "embedding repo. Falls back to the HF_TOKEN env var "
        "or a cached `huggingface-cli login`.",
    )
    # --render and --render-all are mutually exclusive (never both).
    rgrp = parser.add_mutually_exclusive_group()
    rgrp.add_argument(
        "--render",
        action="store_true",
        help="Render the scene(s) at the single baseline config "
        "(512, white, city, 50mm, pitch 0, yaw 0).",
    )
    rgrp.add_argument(
        "--render-all",
        action="store_true",
        help="Render the scene(s) across the 6 factor sweeps "
        "(resolution, focal, pitch, yaw@45, env, background).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose pipeline output (conflicts, depths).",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory for run folders.",
    )
    args = parser.parse_args(argv)

    # Resolve the render mode (None / 'render' / 'render_all').
    if args.render and args.render_all:
        # Mutually-exclusive group already prevents this, but guard anyway.
        parser.error("--render and --render-all are mutually exclusive.")
    render_mode = (
        "render"
        if args.render
        else ("render_all" if args.render_all else None)
    )

    # ---- Path mode: no generation, render an existing run. ----
    if args.path:
        if not render_mode:
            parser.error("--path requires --render or --render-all.")
        run_dir = os.path.abspath(args.path)
        if not os.path.isdir(run_dir):
            parser.error(f"--path is not a directory: {run_dir}")
        if not os.path.isfile(os.path.join(run_dir, "config.json")):
            parser.error(
                f"--path must be a run dir containing config.json: {run_dir}"
            )
        room_dims = _room_dims_from_config(run_dir)
        # Render base + any variant_* dirs present.
        want_variants = any(
            os.path.isdir(os.path.join(run_dir, v))
            for v in (
                "variant_01_half",
                "variant_02_biggest-only",
                "variant_03_scrambled",
                "variant_04_worst-object",
            )
        )
        print(f"[cli] rendering existing run: {run_dir}")
        _render_run(run_dir, room_dims, render_mode, want_variants)
        print(f"[cli] done (render-only).")
        return 0

    # ---- Prompt mode: generate, optionally render. ----
    if not args.api_key:
        print(
            "[cli] WARNING: no API key set (use --api-key or OPENAI_API_KEY).",
            file=sys.stderr,
        )

    run_id = _utc_run_id()
    out_dir = os.path.join(args.outputs_root, run_id)
    base_dir = os.path.join(out_dir, "base")
    os.makedirs(base_dir, exist_ok=True)
    print(f"[cli] run -> {out_dir}")

    # 1. Generate the base scene graph (LLM + backtracking solver).
    llm_config = _build_llm_config(args)
    i_design = _generate_scene(args, llm_config)

    # scene_graph is a flat list (real objects + room priors) after backtrack.
    scene_graph = i_design.scene_graph
    base_scene_path = os.path.join(base_dir, "scene_graph.json")
    with open(base_scene_path, "w") as f:
        json.dump(scene_graph, f, indent=4)
    print(f"[cli] base scene written: {base_scene_path}")

    # Count real objects (exclude room priors) for config + variants.
    from vlmunr_variants import filter_real_objects

    real_objects = filter_real_objects(scene_graph)
    n_real = len(real_objects)

    # 2. Optionally retrieve assets (best match) for the base scene.
    retrieval_used = _maybe_load_retrieval_backend(args)
    if retrieval_used:
        _retrieve_base_assets(base_dir, scene_graph)

    # 3. Persist config.json (prompt + llm config + metadata).
    _write_config(out_dir, args, run_id, n_real, retrieval_used)

    # 4. Optionally produce the four content variants (no regeneration).
    made_variants = False
    if args.variants:
        if n_real == 0:
            print(
                "[cli] no real objects to variant; skipping --variants.",
                file=sys.stderr,
            )
        else:
            _make_variants(out_dir, base_dir, args, list(args.room_dims))
            made_variants = True

    # 5. Optionally render base (+ variants).
    if render_mode:
        _render_run(out_dir, list(args.room_dims), render_mode, made_variants)

    print(f"[cli] done. run_id={run_id} real_objects={n_real}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
