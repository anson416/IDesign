"""CLI: generate an I-Design scene from a textual description.

Runs the full multi-agent I-Design pipeline (designer -> architect -> engineer
-> corrector -> refiner -> backtracking solver) from a free-text room prompt,
then optionally retrieves the 3D assets and writes the four content variants.

Layout
-------
outputs/
  <YYYYMMDD-HHMMSS>/            # datetime.datetime.now(datetime.UTC)
    config.json                  # prompt + llm config + run metadata
    scene_graph.json             # BASE scene (flat list: real objects + priors)
    Assets/                      # best-match .glb per object (if --retrieve)
    variant_01_half/
      scene_graph.json           # keep round(n/2) objects
    variant_02_biggest-only/
      scene_graph.json           # keep the single largest object
    variant_03_scrambled/
      scene_graph.json           # randomize positions within the room
    variant_04_worst-object/
      scene_graph.json           # fork + worst-CLIP assets
      Assets/                    # worst-match .glb per object (if --retrieve)

The base scene_graph.json is always written. Variants are only produced when
--variants is passed (they are cheap JSON transforms / asset swaps of the
already-generated base scene; no LLM regeneration).

Usage
-----
  python scene_cli.py --prompt "A creative vibrant living room" \\
      --model gpt-5.1-2025-11-13 --base-url https://api.chatanywhere.tech/v1 \\
      --api-key $OPENAI_API_KEY --temperature 0.7 [--no-of-objects 15] \\
      [--room-dims 4.0 4.0 2.5] [--seed 42] [--variants] [--retrieve]
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
        "n_real_objects": n_real,
        "llm": {
            "model": args.model,
            "base_url": args.base_url,
            "api_key": "***" if args.api_key else "",
            "temperature": args.temperature,
            "json_model": args.json_model,
        },
        "retrieval_backend_loaded": retrieval_used,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

        retrieve.load_backend()
        return retrieve.backend_available()
    except Exception as e:
        print(f"[cli] retrieval backend unavailable ({e}); "
              "skipping asset download (scenes still written).")
        return False


def _retrieve_base_assets(out_dir, scene_graph):
    """Download the best-match .glb for every real object into Assets/."""
    import retrieve

    assets_dir = os.path.join(out_dir, "Assets")
    result = retrieve.retrieve_scene_assets(
        scene_graph, assets_dir, match="best", sim_th=0.1, verbose=True,
        autoload=False,  # backend already loaded by _maybe_load_retrieval_backend
    )
    print(f"[cli] base assets: placed={len(result['placed'])} "
          f"skipped={len(result['skipped'])}")
    return assets_dir


def _make_variants(out_dir, args, room_dims, scene_dir_for_variants):
    """Generate the four named variants as sibling subfolders of out_dir."""
    import vlmunr_variants as variants

    created = variants.generate_named_variants(
        scene_dir_for_variants,
        seed=args.seed,
        room_dims=room_dims,
        worst_rank=0,
    )
    for name, path in created.items():
        print(f"[cli] variant {name}: {path}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an I-Design scene from a textual description."
    )
    parser.add_argument(
        "--prompt", required=True,
        help="Free-text scene/room description (e.g. 'A cozy reading nook').",
    )
    parser.add_argument("--model", default="gpt-5.1-2025-11-13",
                        help="LLM model name for the chat agents.")
    parser.add_argument("--json-model", default="gpt-4o",
                        help="LLM model name for JSON-mode agent calls "
                             "(engineer + schema debugger).")
    parser.add_argument("--base-url",
                        default=os.environ.get(
                            "OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
                        help="LLM API base URL.")
    parser.add_argument("--api-key",
                        default=os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("CHATANYWHERE_API_KEY", ""),
                        help="LLM API key.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature applied to ALL agents.")
    parser.add_argument("--no-of-objects", type=int, default=15,
                        help="Target number of objects in the scene.")
    parser.add_argument("--room-dims", nargs=3, type=float,
                        default=[4.0, 4.0, 2.5],
                        metavar=("LENGTH", "WIDTH", "HEIGHT"),
                        help="Room dimensions in meters (x, y, z).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic variants.")
    parser.add_argument("--variants", action="store_true",
                        help="Also write the four content variants "
                             "(variant_01_half, variant_02_biggest-only, "
                             "variant_03_scrambled, variant_04_worst-object).")
    parser.add_argument("--retrieve", action="store_true",
                        help="Download 3D assets (best match for base, worst "
                             "match for variant_04) via OpenShape/CLIP. Needs "
                             "GPU + ~3GB embeddings; degrades gracefully.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose pipeline output (conflicts, depths).")
    parser.add_argument("--outputs-root", default="outputs",
                        help="Root directory for run folders.")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("[cli] WARNING: no API key set (use --api-key or OPENAI_API_KEY).",
              file=sys.stderr)

    run_id = _utc_run_id()
    out_dir = os.path.join(args.outputs_root, run_id)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[cli] run -> {out_dir}")

    # 1. Generate the base scene graph (LLM + backtracking solver).
    llm_config = _build_llm_config(args)
    i_design = _generate_scene(args, llm_config)

    # scene_graph is a flat list (real objects + room priors) after backtrack.
    scene_graph = i_design.scene_graph
    base_scene_path = os.path.join(out_dir, "scene_graph.json")
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
        _retrieve_base_assets(out_dir, scene_graph)

    # 3. Persist config.json (prompt + llm config + metadata).
    _write_config(out_dir, args, run_id, n_real, retrieval_used)

    # 4. Optionally produce the four content variants (no regeneration).
    if args.variants:
        if n_real == 0:
            print("[cli] no real objects to variant; skipping --variants.",
                  file=sys.stderr)
        else:
            _make_variants(out_dir, args, list(args.room_dims), out_dir)

    print(f"[cli] done. run_id={run_id} real_objects={n_real}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
