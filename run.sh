#!/usr/bin/env bash
# run.sh -- audit driver: for each scene dir, generate content variants then
# render all audit phases for the original scene and every variant.
#
# Usage:
#   ./run.sh scenes_root [phase]
#
# scenes_root: directory containing one or more scene dirs (each with
#              scene_graph.json + Assets/).
# phase:       one of 1a|1b|1c|1d|2|all (default: all).
#
# Renderings are written to <scene-dir>/renderings/. Variant scene dirs are
# created as siblings (<scene>_variant_half, etc.); variant renderers read
# Assets from the original scene via the recorded assets-dir marker.

set -euo pipefail

SCENES_ROOT="${1:?usage: run.sh <scenes_root> [phase]}"
PHASE="${2:-all}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

for scene_dir in "$SCENES_ROOT"/*/; do
  scene_dir="${scene_dir%/}"
  [ -f "$scene_dir/scene_graph.json" ] || continue
  # Skip directories that are themselves variants.
  case "$scene_dir" in
    *_variant_*) continue ;;
  esac

  echo "=== scene: $scene_dir ==="

  echo "--- generating variants ---"
  "$PYTHON" "$HERE/vlmunr_variants.py" --scene-dir "$scene_dir" --seed "$SEED"

  echo "--- rendering original (phase $PHASE) ---"
  "$PYTHON" "$HERE/vlmunr_render.py" --scene-dir "$scene_dir" --phase "$PHASE"

  for variant in "${scene_dir}"_variant_*; do
    [ -d "$variant" ] || continue
    echo "--- rendering $variant (phase $PHASE) ---"
    "$PYTHON" "$HERE/vlmunr_render.py" --scene-dir "$variant" --phase "$PHASE"
  done
done

echo "[run] done."
