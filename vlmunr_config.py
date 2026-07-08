"""Factor levels, baseline, and paths for the rendering audit harness.

This module is import-light (pure Python, no bpy / torch) so it can be used by
unit tests and CLI argument parsing without a Blender runtime.

The render factors match the user's spec EXACTLY:

  (1) resolution sweep  : [196, 224, 256, 336, 384, 448, 512, 768, 1024]
  (2) focal-length sweep: [16, 24, 35, 50, 85, 100, 200]
  (3) pitch sweep       : [0, 15, 30, 45, 60, 75, 90]   (yaw held at 0)
  (4) yaw sweep         : [0, 45, 90, 135, 180, 225, 270, 315]  (pitch held at 45)
  (5) env-map sweep     : [city, courtyard, forest, interior, night, studio,
                            sunrise, sunset]
  (6) background sweep  : 10 colors (6 grays + white + 3 chromatic)

Baseline (held for any factor NOT being swept):
  res=512, focal=50, bg=(255,255,255), env=city, pitch=0, yaw=0
Exception: the yaw sweep runs at pitch=45 (not the baseline pitch 0), so the
oblique camera sees into the room and the azimuth is a real viewpoint change
rather than a trivial near-wall occlusion.

`--render`   -> the single baseline config (the one above).
`--render-all` -> the deduped union of the 6 sweeps above.
"""

import os

# Directory holding the HDRI environment maps (siblings of this module).
HDRI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmunr_hdri")

# ----------------------------------------------------------------------------
# Factor levels swept by the audit (the user's exact spec).
# ----------------------------------------------------------------------------
RESOLUTIONS = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
FOCAL_LENGTHS = [16, 24, 35, 50, 85, 100, 200]
PITCHES = [0, 15, 30, 45, 60, 75, 90]
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]
HDRIS = [
    "city",
    "courtyard",
    "forest",
    "interior",
    "night",
    "studio",
    "sunrise",
    "sunset",
]
# Background colors: 6 grays + white + 3 chromatic (red/green/blue). Each is an
# (r, g, b) byte triple; the renderer composites it over the transparent
# master via PIL alpha-composite.
BACKGROUNDS = [
    (0, 0, 0),
    (65, 65, 65),
    (118, 118, 118),
    (128, 128, 128),
    (186, 186, 186),
    (204, 204, 204),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

# ----------------------------------------------------------------------------
# Baseline (held constant for any factor not being swept).
# ----------------------------------------------------------------------------
BASELINE_RES = 512
BASELINE_FOCAL = 50
BASELINE_BG = (255, 255, 255)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0
# The yaw (azimuth) sweep runs at a fixed pitch of 45 degrees, NOT the
# top-down baseline pitch of 0. At pitch 0 the camera looks straight down
# and azimuth is meaningless; at pitch 45 the near walls are culled and the
# azimuth becomes a genuine viewpoint manipulation.
YAW_SWEEP_PITCH = 45

# In the bpa render convention, pitch == 0 is top-down (camera looks straight
# down the -Z axis). Filename pitch is always the bpa pitch value as-is.
PHASES = ["1a", "1d", "2_pitch", "2_yaw", "1c", "1b"]


def hdri_path(hdri: str) -> str:
    """Absolute path to the .exr file for an HDRI name."""
    return os.path.join(HDRI_DIR, f"{hdri}.exr")


def _baseline_config(**overrides) -> dict:
    """A fully-resolved render config; overrides replace baseline factors.

    Each config dict has keys: res, focal, bg, hdri, pitch, yaw.
    """
    cfg = {
        "res": BASELINE_RES,
        "focal": BASELINE_FOCAL,
        "bg": BASELINE_BG,
        "hdri": BASELINE_HDRI,
        "pitch": BASELINE_PITCH,
        "yaw": BASELINE_YAW,
    }
    cfg.update(overrides)
    return cfg


def BASELINE_CONFIG() -> dict:
    """The single baseline render config (the `--render` target).

    res=512, focal=50, bg=(255,255,255), hdri=city, pitch=0, yaw=0.
    """
    return _baseline_config()


def single_render_config() -> list[dict]:
    """The one config rendered by `--render` (the baseline)."""
    return [BASELINE_CONFIG()]


def phase_levels(phase: str) -> list[dict]:
    """Return the swept config list for one of the 6 phases.

    Each entry is a fully-resolved config dict (keys: res, focal, bg, hdri,
    pitch, yaw). All factors not swept by the phase are held at the baseline,
    EXCEPT the yaw phase which runs at pitch=45 (YAW_SWEEP_PITCH).

        phase '1a'      -> sweep resolutions
        phase '1d'      -> sweep focal lengths
        phase '2_pitch' -> sweep pitch at baseline yaw (0)
        phase '2_yaw'   -> sweep yaw at fixed pitch YAW_SWEEP_PITCH (45)
        phase '1c'      -> sweep HDRIs
        phase '1b'      -> sweep background colors
    """
    if phase == "1a":
        return [_baseline_config(res=r) for r in RESOLUTIONS]
    if phase == "1d":
        return [_baseline_config(focal=f) for f in FOCAL_LENGTHS]
    if phase == "2_pitch":
        return [_baseline_config(pitch=p, yaw=BASELINE_YAW) for p in PITCHES]
    if phase == "2_yaw":
        return [_baseline_config(pitch=YAW_SWEEP_PITCH, yaw=y) for y in YAWS]
    if phase == "1c":
        return [_baseline_config(hdri=h) for h in HDRIS]
    if phase == "1b":
        return [_baseline_config(bg=c) for c in BACKGROUNDS]
    raise ValueError(f"Unknown phase: {phase!r}")


def all_render_configs() -> list[dict]:
    """The deduped union of all 6 phase sweeps (the `--render-all` target).

    Configs are deduped by (res, focal, bg, hdri, pitch, yaw) so the shared
    baseline point (which belongs to every sweep) is rendered once. Returns the
    configs in a stable order: phase-by-phase, each phase in its factor order,
    skipping already-seen configs.
    """
    configs: list[dict] = []
    seen: set = set()
    for phase in PHASES:
        for c in phase_levels(phase):
            key = (
                c["res"], c["focal"], tuple(c["bg"]),
                c["hdri"], c["pitch"], c["yaw"],
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(c)
    return configs


# Backward-compat: the standalone `vlmunr_render.py --phase all` still works.
ALL_PHASES = list(PHASES)
