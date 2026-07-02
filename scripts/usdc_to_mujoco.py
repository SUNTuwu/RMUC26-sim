#!/usr/bin/env python3
"""
Convert USD scene to OBJ meshes + generate MuJoCo XML include.
Usage: python3 usdc_to_mujoco.py <input.usdc> <output_dir>
"""
import sys
import os
import json
from pxr import Usd, UsdGeom, Gf

def usdc_to_mujoco(usdc_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    stage = Usd.Stage.Open(usdc_path)
    if not stage:
        print(f"ERROR: Cannot open {usdc_path}")
        return

    meshes = []
    bodies = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points_attr = mesh.GetPointsAttr()
        faces_attr = mesh.GetFaceVertexIndicesAttr()
        counts_attr = mesh.GetFaceVertexCountsAttr()

        points = points_attr.Get()
        faces = faces_attr.Get()
        counts = counts_attr.Get()

        if not points or not faces:
            continue

        name = prim.GetName()
        path = str(prim.GetPath())

        # Get world transform
        xform = UsdGeom.Xformable(prim)
        world_mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_mat.ExtractTranslation()
        rotation = world_mat.ExtractRotation()
        euler = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))

        # Write OBJ
        obj_name = f"{name}.obj"
        obj_path = os.path.join(out_dir, obj_name)
        with open(obj_path, 'w') as f:
            f.write(f"# {name} from {path}\n")
            f.write(f"mtllib {name}.mtl\n")
            f.write(f"o {name}\n")
            for p in points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

            idx = 0
            for cnt in counts:
                f.write("f")
                for _ in range(cnt):
                    f.write(f" {faces[idx] + 1}")
                    idx += 1
                f.write("\n")

        meshes.append({
            "name": name,
            "obj": obj_name,
            "path": path,
            "verts": len(points),
            "faces": len(counts),
            "pos": [translation[0], translation[1], translation[2]],
            "euler": [euler[0], euler[1], euler[2]],  # degrees
        })
        print(f"  {name}: {len(points)} verts, {len(counts)} faces -> {obj_name}")

    # Write manifest
    manifest = {
        "usdc": usdc_path,
        "output_dir": out_dir,
        "meshes": meshes,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal: {len(meshes)} meshes exported to {out_dir}/")
    print(f"Manifest: {manifest_path}")

    # Generate MuJoCo asset XML snippet
    xml_snippet = '  <!-- Auto-generated from USD scene -->\n'
    for m in meshes:
        xml_snippet += f'  <mesh name="{m["name"]}" file="{m["obj"]}"/>\n'
    xml_snippet += '\n  <!-- Bodies in worldbody -->\n'
    for m in meshes:
        p = m['pos']
        e = m['euler']
        xml_snippet += f'  <body name="{m["name"]}" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" euler="{e[0]:.2f} {e[1]:.2f} {e[2]:.2f}">\n'
        xml_snippet += f'    <geom name="{m["name"]}_geom" type="mesh" mesh="{m["name"]}" rgba="0.6 0.6 0.7 1.0"/>\n'
        xml_snippet += f'  </body>\n'

    snippet_path = os.path.join(out_dir, "scene_assets.xml")
    with open(snippet_path, 'w') as f:
        f.write(xml_snippet)
    print(f"MJCF snippet: {snippet_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 usdc_to_mujoco.py <input.usdc> [output_dir]")
        sys.exit(1)

    usdc_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(usdc_path)[0] + "_meshes"
    usdc_to_mujoco(usdc_path, out_dir)
