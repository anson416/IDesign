"""vlmunr_shell.py — dollhouse room-shell builder for the audit renderers.

Builds a neutral floor + 4 walls and tags the walls so they can be back-face
culled per camera (near walls hidden at oblique views; all shown at top-down).

Coordinate convention: Z-up, meters, floor at z=0. Camera convention matches
vlmunr_bpa.render_perspective: camera sits at
    cam = center + R(pitch,0,yaw) @ (0,0,1) * distance
so the camera's horizontal direction from center is the XY part of that vector.

The wall's OUTWARD normal is stored on the object; a wall is hidden when its
outward face points toward the camera (it's a near wall).
"""
import math

NEUTRAL_RGB = (0.72, 0.72, 0.72)   # matte neutral shell material


def _neutral_material(bpy, name="vlmunr_shell_mat"):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (*NEUTRAL_RGB, 1.0)
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.9
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = 0.2
            elif "Specular" in bsdf.inputs:
                bsdf.inputs["Specular"].default_value = 0.2
    return m


def _add_quad(bpy, name, verts):
    """Create a single-quad mesh object from 4 world-space verts."""
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    mesh.from_pydata([tuple(v) for v in verts], [], [[0, 1, 2, 3]])
    mesh.update()
    return obj


def build_room_shell(bpy, floor_verts, wall_height, *, thickness=0.0,
                     margin=0.0, ceiling=False):
    """Build floor + walls from a floor polygon (list of (x,y[,z]) in meters,
    CCW or CW) and a wall height. Returns list of wall objects (tagged).

    floor_verts: polygon corners at z=0 (z ignored if given).
    """
    mat = _neutral_material(bpy)

    pts = [(float(v[0]), float(v[1])) for v in floor_verts]
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    # optional margin: push each corner outward from centroid
    if margin:
        expanded = []
        for x, y in pts:
            dx, dy = x - cx, y - cy
            d = math.hypot(dx, dy) or 1.0
            expanded.append((x + dx / d * margin, y + dy / d * margin))
        pts = expanded

    # floor: ngon at z=0
    floor = _add_quad(bpy, "vlmunr_floor",
                      [(x, y, 0.0) for x, y in pts]) if n == 4 else None
    if floor is None:
        mesh = bpy.data.meshes.new("vlmunr_floor")
        floor = bpy.data.objects.new("vlmunr_floor", mesh)
        bpy.context.scene.collection.objects.link(floor)
        mesh.from_pydata([(x, y, 0.0) for x, y in pts], [], [list(range(n))])
        mesh.update()
    floor.data.materials.append(mat)

    if ceiling:
        cobj = _add_quad(bpy, "vlmunr_ceiling",
                         [(x, y, wall_height) for x, y in pts]) if n == 4 else None
        if cobj is not None:
            cobj.data.materials.append(mat)

    walls = []
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        # wall quad: bottom edge (x0,y0)->(x1,y1), up to wall_height
        w = _add_quad(bpy, f"vlmunr_wall_{i}", [
            (x0, y0, 0.0), (x1, y1, 0.0),
            (x1, y1, wall_height), (x0, y0, wall_height),
        ])
        w.data.materials.append(mat)
        # inward normal (points toward centroid): edge midpoint -> centroid
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ix, iy = cx - mx, cy - my
        d = math.hypot(ix, iy) or 1.0
        ix, iy = ix / d, iy / d
        w["VLMUNR_WALL"] = 1
        w["VLMUNR_OX"], w["VLMUNR_OY"] = -ix, -iy   # OUTWARD normal
        walls.append(w)
    return walls


def cull_walls(walls, pitch_deg, yaw_deg, *, top_down_thresh=10.0, dot_thresh=0.15):
    """Hide near walls for a dollhouse view. pitch=0 => top-down (walls edge-on,
    show all). At oblique pitch, hide walls whose outward normal faces the
    camera. `rotation=(pitch,0,yaw)`; camera dir from center = R@(0,0,1)."""
    try:
        from mathutils import Euler, Vector
    except Exception:
        return
    # camera horizontal direction from center
    R = Euler((math.radians(pitch_deg), 0.0, math.radians(yaw_deg))).to_matrix()
    cam = R @ Vector((0.0, 0.0, 1.0))
    cdx, cdy = cam.x, cam.y
    horiz = math.hypot(cdx, cdy)
    top_down = (pitch_deg <= top_down_thresh) or (horiz < 1e-3)
    if horiz > 1e-6:
        cdx, cdy = cdx / horiz, cdy / horiz
    for w in walls:
        if top_down:
            w.hide_render = False
            continue
        ox, oy = w["VLMUNR_OX"], w["VLMUNR_OY"]
        w.hide_render = (ox * cdx + oy * cdy) > dot_thresh
