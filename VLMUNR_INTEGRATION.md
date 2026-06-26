# VLM-Unreliability Audit Integration (I-Design)

A self-contained rendering + content-variant layer added to the I-Design repo
so its generated 3D scenes can be fed into a VLM-evaluator audit harness. It
sweeps controlled rendering factors (resolution, focal length, background,
HDRI, camera pitch/yaw) and produces content variants (object removal,
worst-match asset substitution).

Nothing here imports an external `vlmunr` package; the renderer (`vlmunr_bpa.py`)
and HDRI maps are copied in.

## Files added

| File | Purpose |
|---|---|
| `vlmunr_bpa.py` | Tested headless Blender renderer (`Builder`, `Renderer`, `clear`, `initialize`, `import_obj`, `transform`). Copied verbatim from the vlm-unreliability repo. |
| `vlmunr_hdri/*.exr` | 8 HDRI environment maps: city, courtyard, forest, interior, night, studio, sunrise, sunset (+ `license.txt`). |
| `vlmunr_config.py` | Factor levels, baseline, paths, and `phase_levels(phase)`. Pure Python (no bpy/torch). |
| `vlmunr_render.py` | CLI renderer. Loads a scene, builds it in Blender with the exact I-Design transform, sweeps factors, writes PNGs. Filename builders + scene iteration are pure functions. |
| `vlmunr_variants.py` | CLI content-variant generator (3 removal + 3 worst-match variants). Filter/removal logic is pure and unit-tested. |
| `vlmunr_smoke.py` | Standalone synthetic-cube smoke render (used by the test suite via subprocess). |
| `tests/test_vlmunr_integration.py` | pytest: filename builders, room-prior filtering, removal fractions, transform math, config phases, bpy smoke render. |
| `gen.sh` | Thin driver documenting/looping the scene-generation step. |
| `run.sh` | Driver looping scene dirs: generate variants, then render all phases for the original + each variant. |

## Run commands

```bash
PY=/Users/anson/miniforge3/envs/vlmunr/bin/python

# Generate the 6 content variants as sibling dirs of the scene.
$PY vlmunr_variants.py --scene-dir path/to/scene [--seed 42]

# Render all factor phases (or a single phase: 1a|1b|1c|1d|2).
$PY vlmunr_render.py --scene-dir path/to/scene --phase all

# Batch: generate variants + render everything for all scenes under a root.
./run.sh scenes_root [phase]

# Scene generation (needs I-Design API + GPU; not exercised by tests).
./gen.sh prompts.txt scenes_root
```

A scene dir must contain `scene_graph.json` (flat I-Design list) and `Assets/<id>.glb`.

## Factor levels

- `RESOLUTIONS = [224, 256, 384, 448, 512, 640, 768, 1024]`
- `FOCAL_LENGTHS = [24, 35, 50, 85, 100, 200]`
- `BACKGROUND_GRAYS = [0, 18, 65, 117, 128, 186, 204, 255]` (used as `(g,g,g)`)
- `HDRIS = [city, courtyard, forest, interior, night, studio, sunrise, sunset]`
- `PITCHES = [0, 30, 60, 90]` (pitch 0 == top-down in the bpa convention)
- `YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]`

Baseline (held for any non-swept factor): RES=512, FOCAL=50, BG=(128,128,128),
HDRI=city, PITCH=0, YAW=0.

Phases (`phase_levels(phase)`):

| Phase | Sweeps |
|---|---|
| 1a | resolutions |
| 1b | background grays |
| 1c | HDRIs |
| 1d | focal lengths |
| 2  | pitch x yaw (cartesian product) |

## Filename scheme

Two-phase rendering per `(res, focal, pitch, yaw, hdri)`:

1. Transparent master:
   `render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png`
2. Background composited (one per gray) via `Renderer.add_bg_to_rgba`:
   `render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png`

All PNGs are written to `<scene-dir>/renderings/`. Changing HDRI triggers
`initialize(environment_map=("vlmunr_hdri/<hdri>.exr", 1.0))`.

## I-Design coordinate convention

`scene_graph.json` is a flat JSON **list**. Real objects have `new_object_id`,
`position` {x,y,z} (meters, z = base height), `rotation` {z_angle: degrees},
`size_in_meters` {length(x), width(y), height(z)}. Room-prior entries carry an
`itemType` field (wall/floor/ceiling) and ids
`south_wall/north_wall/east_wall/west_wall/"middle of the room"/ceiling`; these
are **filtered out**.

Per-object Blender transform (replicated EXACTLY from `place_in_blender.py`):

1. Import `Assets/<id>.glb`, join child meshes into one, origin -> BOUNDS.
2. `location = (position.x, position.y, position.z)`.
3. Rotate about world Z by `(z_angle / 180) * pi + pi` radians.
   **Note the `+pi` (+180deg) offset** — it is intentional and matches upstream.
4. Rescale so the object's bounding-box dimensions equal `(length, width, height)`.

Z-up, meters. Computed as pure values by `vlmunr_render.idesign_object_transform`.

### Variants

- Removal (`variant_half/quarter/eighth`): keep `round(n/2 | n/4 | n/8)` real
  objects (always >= 1) via seeded `random.sample`; deterministic per seed.
- Worst-match (`variant_alt_0/2/4`): structured to swap each object's asset for a
  low-CLIP-ranked retrieval result (rank 0/2/4 from the worst end) through a
  lazy-imported retrieval hook (`vlmunr_retrieval_hook.retrieve_worst_match`,
  not present). **Degrades gracefully**: if retrieval is unavailable, the scene
  is copied unchanged and the intent is recorded as a leading
  `{"_vlmunr_alt_intent": {object_id: rank}}` entry (ignored by the renderer).

Variant dirs are siblings (`<scene>_variant_*`); each records the original
`Assets/` path in `vlmunr_assets_dir.txt`, so the renderer reads shared assets
from the original scene (no copying). `--assets-dir` overrides this.

## VERIFICATION STATUS

### Verified (unit + synthetic bpy, no real assets)

- All filename builders produce exact expected strings.
- Room-prior filtering (by `itemType` and by known id) drops non-objects.
- Removal fractions: kept counts equal `round(n/2|4|8)`, clamped `>= 1`,
  deterministic per seed, differ across seeds.
- Worst-match generator degrades gracefully and records intent when retrieval
  is unavailable; renderer still iterates the correct real objects.
- `generate_variants` writes all 6 variant dirs with valid `scene_graph.json`.
- I-Design transform math (location, dims, the `+180deg` Z-rotation offset) as
  pure functions.
- `phase_levels` counts and baseline-holding for every phase.
- **bpy smoke render** (`vlmunr_smoke.py`, run as a subprocess by the test): a
  synthetic 2-cube scene renders ONE config to a non-empty master PNG and a
  non-empty background composite.
- **End-to-end synthetic glb path** (manual check): primitive cubes exported to
  `.glb`, imported + joined + transformed via `load_scene_into_blender`, then a
  full `render_phases("1b")` sweep (1 master + 8 composites, all non-empty),
  including a variant dir rendering with assets resolved from the original via
  the marker file.

Test result: `16 passed`.

### Requires real assets / API to validate (NOT verified here)

- **True coordinate correctness on real Objaverse assets** — the transform math
  matches `place_in_blender.py` and runs on synthetic glb, but exact placement
  fidelity on real I-Design `Assets/*.glb` is not asserted (no real assets
  downloaded).
- **Worst-match retrieval** — the rank-based CLIP/Objaverse substitution
  (`retrieve.py` path) needs a GPU plus model/asset downloads; only the
  graceful-degradation branch is exercised.
- **Full real-scene factor sweep** at production resolutions/HDRIs (only a
  reduced synthetic sweep was run).
- **`gen.sh` scene generation** — depends on I-Design's LLM API + GPU; documented
  but not executed.
