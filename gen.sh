#!/usr/bin/env bash
# gen.sh -- scene generation driver for the VLM-unreliability audit.
#
# I-Design's test.py hardcodes a single prompt, so true batch generation
# requires editing test.py per prompt (it calls the OpenAI-backed IDesign
# pipeline and writes scene_graph.json + retrieves Assets/ via retrieve.py).
#
# This driver documents the generation step and loops over a prompts file,
# producing one scene directory per prompt. It is a THIN wrapper: it assumes
# you have configured I-Design's API credentials and that generation writes a
# scene_graph.json into the CWD (as upstream test.py does).
#
# Usage:
#   ./gen.sh prompts.txt scenes_root
#
# prompts.txt: one free-text room prompt per line.
# scenes_root: directory under which scene_<n>/ subdirs are created.
#
# NOTE: generation needs network + model APIs + GPU for retrieve.py; it is NOT
# exercised by the unit/smoke tests.

set -euo pipefail

PROMPTS_FILE="${1:?usage: gen.sh <prompts.txt> <scenes_root>}"
SCENES_ROOT="${2:?usage: gen.sh <prompts.txt> <scenes_root>}"
PYTHON="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"

mkdir -p "$SCENES_ROOT"

i=0
while IFS= read -r prompt; do
  [ -z "$prompt" ] && continue
  scene_dir="$SCENES_ROOT/scene_$i"
  mkdir -p "$scene_dir"
  echo "[gen] scene_$i: $prompt"
  # I-Design generation runs in its CWD and emits scene_graph.json there.
  # Upstream test.py hardcodes the prompt; to parametrize, set IDESIGN_PROMPT
  # and adapt test.py to read it, or generate manually then drop the resulting
  # scene_graph.json + Assets/ into "$scene_dir".
  (
    cd "$scene_dir"
    IDESIGN_PROMPT="$prompt" "$PYTHON" "$(dirname "$0")/test.py" || {
      echo "[gen] generation for scene_$i requires I-Design API/GPU; skipping" >&2
    }
  )
  i=$((i + 1))
done < "$PROMPTS_FILE"

echo "[gen] done: generated up to $i scene dir(s) under $SCENES_ROOT"
