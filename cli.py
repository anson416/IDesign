"""cli.py — I-Design scene generation from a textual description.

This is the front entry point for the I-Design method. It runs the full
multi-agent LLM pipeline (Interior Designer -> Interior Architect -> Engineer
-> Layout Corrector -> Layout Refiner -> backtracking solver) from a free-text
room prompt, writes the base scene, and optionally retrieves 3D assets and
produces the four content variants.

It is a thin wrapper over `scene_cli.py`; it adds no behaviour, only a
convenient name and a thorough setup guide.

==============================================================================
EXTERNAL LOCAL RESOURCES THIS METHOD REQUIRES
==============================================================================

The method has TWO stages with very different resource needs:

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

  6. (OPTIONAL) Blender / `bpy` — NOT required by this CLI.
     - Needed only by vlmunr_render.py to turn a scene + Assets/ into PNGs.
     This CLI stops at scene_graph.json + Assets/ + variants. To render, run
     `python vlmunr_render.py --scene-dir <run-dir>` from a Blender-bundled
     Python (bpy is not in the plain idesign env).

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

Output layout (under outputs/<YYYYMMDD-HHMMSS-UTC>/):
  config.json                      prompt + llm config (api key masked) + metadata
  base/
    scene_graph.json               BASE scene (flat list: real objects + room priors)
    Assets/                        best-match .glb per object (with --retrieve)
    renderings/                    (if rendered) ALWAYS named "renderings"
  variant_01_half/                 keep round(n/2) objects (seeded, no regen)
  variant_02_biggest-only/         keep the single largest object (by volume)
  variant_03_scrambled/            randomize positions within the room (no regen)
  variant_04_worst-object/         fork + worst-CLIP asset per object (own Assets/)

Variants are cheap JSON transforms / asset swaps of the already-generated
base scene — they do NOT re-run the LLM (no extra cost). variant_04 reuses the
base asset and records "worst_fallback" intent when retrieval is unavailable.
"""

import sys

# This module is a thin docstring + arg-parsing wrapper around scene_cli.main.
import scene_cli


def main(argv=None):
    # Reuse the full argument parser / pipeline from scene_cli.py so there is
    # exactly one source of truth for flags and behaviour.
    return scene_cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
