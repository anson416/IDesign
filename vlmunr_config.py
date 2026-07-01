"""Factor levels, baseline, and paths for the VLM-unreliability audit harness.

This module is import-light (pure Python, no bpy / torch) so it can be used by
unit tests and CLI argument parsing without a Blender runtime.
"""

import os
from typing import Literal

# Directory holding the copied HDRI environment maps (siblings of this module).
HDRI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmunr_hdri")

# ----------------------------------------------------------------------------
# Factor levels swept by the audit. These match Table 1 of the paper EXACTLY.
# ----------------------------------------------------------------------------
RESOLUTIONS = [196, 224, 256, 336, 384, 448, 512, 768, 1024]
FOCAL_LENGTHS = [16, 24, 35, 50, 85, 100, 200]
BACKGROUND_GRAYS = [0, 65, 128, 186, 204, 255]
# Chromatic (saturated solid) backgrounds: red / green / blue.
BACKGROUND_CHROMATIC = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
# The paper's 4th background condition is a neutral floor-texture render-path
# treatment. It is out of scope for this harness (it is a render-path concern,
# not a flat composite color), so it is exposed only as a documented sentinel
# and is NEVER rendered here.
FLOOR_TEXTURE_BACKGROUND = "floor_texture"
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
# In the bpa render convention, pitch == 0 is top-down (camera looks straight
# down the -Z axis).
PITCHES = [0, 15, 30, 45, 60, 75, 90]
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]

# ----------------------------------------------------------------------------
# Baseline (held constant for any factor not being swept).
# ----------------------------------------------------------------------------
BASELINE_RES = 512
BASELINE_FOCAL = 50
BASELINE_BG = (128, 128, 128)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0
# Per Table 1, the yaw (azimuth) sweep is run at a fixed pitch of 45 degrees,
# not at the top-down baseline pitch of 0.
BASELINE_YAW_PITCH = 45

Phase = Literal[
    "1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"
]


def hdri_path(hdri: str) -> str:
    """Absolute path to the .exr file for an HDRI name."""
    return os.path.join(HDRI_DIR, f"{hdri}.exr")


def phase_levels(phase: str) -> list[dict]:
    """Return the swept config list for a phase.

    Each entry is a fully-resolved config dict with keys:
        res, focal, bg, hdri, pitch, yaw

    All factors not being swept by the phase are held at the baseline.

        phase '1a'        -> sweep resolutions
        phase '1b'        -> sweep background grays
        phase '1b_chroma' -> sweep chromatic (red/green/blue) backgrounds
        phase '1c'        -> sweep HDRIs
        phase '1d'        -> sweep focal lengths
        phase '2_pitch'   -> sweep pitch at baseline yaw (0)
        phase '2_yaw'     -> sweep yaw at fixed pitch BASELINE_YAW_PITCH (45)
        phase '2'         -> sweep pitch x yaw (cartesian product; kept for
                             backward compatibility)
    """

    def base(**overrides) -> dict:
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

    if phase == "1a":
        return [base(res=r) for r in RESOLUTIONS]
    if phase == "1b":
        return [base(bg=(g, g, g)) for g in BACKGROUND_GRAYS]
    if phase == "1b_chroma":
        return [base(bg=c) for c in BACKGROUND_CHROMATIC]
    if phase == "1c":
        return [base(hdri=h) for h in HDRIS]
    if phase == "1d":
        return [base(focal=f) for f in FOCAL_LENGTHS]
    if phase == "2_pitch":
        return [base(pitch=p, yaw=BASELINE_YAW) for p in PITCHES]
    if phase == "2_yaw":
        return [base(pitch=BASELINE_YAW_PITCH, yaw=y) for y in YAWS]
    if phase == "2":
        return [
            base(pitch=p, yaw=y) for p in PITCHES for y in YAWS
        ]
    raise ValueError(f"Unknown phase: {phase!r}")


ALL_PHASES = ["1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"]
