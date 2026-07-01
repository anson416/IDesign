"""Standalone synthetic bpy smoke render for the audit harness.

Builds a 2-cube synthetic scene with vlmunr_bpa primitives (no real assets) and
renders ONE master + composite config. Run as a subprocess so vlmunr_bpa's
stdout-fd redirection never interferes with a parent test runner's capture.

Usage:
    python vlmunr_smoke.py <output_dir>

Prints "SMOKE_OK <master> <composite>" on success and exits 0.
"""

import os
import sys

import vlmunr_bpa as bpa
import vlmunr_config as cfg
import vlmunr_render as render


def run(out_root: str) -> tuple[str, str]:
    os.makedirs(out_root, exist_ok=True)
    bpa.clear()

    # Two primitive cubes as stand-ins for real assets.
    for i, loc in enumerate([(0.0, 0.0, 0.5), (1.5, 0.0, 0.5)]):
        cube = bpa.Builder.new_cube(name=f"cube_{i}")
        bpa.transform(cube, position=loc, scale=(0.8, 0.8, 1.0))

    bpa.initialize(
        transparent=True,
        environment_map=(cfg.hdri_path("city"), 1.0),
    )

    renderer = bpa.Renderer()
    center, radius = renderer.compute_bounding_sphere()

    master = os.path.join(out_root, render.master_filename(224, 50, 0, 0, "city"))
    ok = renderer.render_perspective(
        master,
        center,
        radius,
        rotation=(0, 0, 0),
        resolution=224,
        focal_length=50,
        background=None,
    )
    if not ok or not os.path.exists(master) or os.path.getsize(master) == 0:
        raise RuntimeError("master render failed")

    comp = os.path.join(
        out_root, render.composite_filename(224, 50, (128, 128, 128), 0, 0, "city")
    )
    renderer.add_bg_to_rgba(master, comp, color=(128, 128, 128))
    if not os.path.exists(comp) or os.path.getsize(comp) == 0:
        raise RuntimeError("composite failed")

    return master, comp


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "renderings"
    m, c = run(out_dir)
    print(f"SMOKE_OK {m} {c}")
