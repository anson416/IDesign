"""Factor levels, baseline, and paths for the VLM-unreliability audit harness.

This module is import-light (pure Python, no bpy / torch) so it can be used by
unit tests and CLI argument parsing without a Blender runtime.
"""

import os
from typing import Literal

# Directory holding the copied HDRI environment maps (siblings of this module).
HDRI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmunr_hdri")

# ----------------------------------------------------------------------------
# Factor levels swept by the audit.
# ----------------------------------------------------------------------------
RESOLUTIONS = [224, 256, 384, 448, 512, 640, 768, 1024]
FOCAL_LENGTHS = [24, 35, 50, 85, 100, 200]
BACKGROUND_GRAYS = [0, 18, 65, 117, 128, 186, 204, 255]
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
PITCHES = [0, 30, 60, 90]
YAWS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# ----------------------------------------------------------------------------
# Baseline (held constant for any factor not being swept).
# ----------------------------------------------------------------------------
BASELINE_RES = 512
BASELINE_FOCAL = 50
BASELINE_BG = (128, 128, 128)
BASELINE_HDRI = "city"
BASELINE_PITCH = 0
BASELINE_YAW = 0

Phase = Literal["1a", "1b", "1c", "1d", "2"]


def hdri_path(hdri: str) -> str:
    """Absolute path to the .exr file for an HDRI name."""
    return os.path.join(HDRI_DIR, f"{hdri}.exr")


def phase_levels(phase: str) -> list[dict]:
    """Return the swept config list for a phase.

    Each entry is a fully-resolved config dict with keys:
        res, focal, bg, hdri, pitch, yaw

    All factors not being swept by the phase are held at the baseline.

        phase '1a' -> sweep resolutions
        phase '1b' -> sweep background grays
        phase '1c' -> sweep HDRIs
        phase '1d' -> sweep focal lengths
        phase '2'  -> sweep pitch x yaw (cartesian product)
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
    if phase == "1c":
        return [base(hdri=h) for h in HDRIS]
    if phase == "1d":
        return [base(focal=f) for f in FOCAL_LENGTHS]
    if phase == "2":
        return [
            base(pitch=p, yaw=y) for p in PITCHES for y in YAWS
        ]
    raise ValueError(f"Unknown phase: {phase!r}")


ALL_PHASES = ["1a", "1b", "1c", "1d", "2"]
