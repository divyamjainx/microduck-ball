# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild the Microduck robot and ball USD assets.

The robot model and meshes come from a pinned commit of Pollen Robotics'
Microduck RL repository (https://github.com/pollen-robotics/microduck_rl,
Apache-2.0). Use --local to supply a local copy. Conversion requires a
compatible Isaac Lab checkout.

The builder uses the bundled basketball when available and falls back to
NVIDIA ArchVis Ball_Walnut, downloading it only if needed. Use --ball to
select a different visual mesh. A sphere collider handles physical contact.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

REPO = "pollen-robotics/microduck_rl"
COMMIT = "b02fdea4902c2b93b7d34a83a6f05da4ececb7d3"
"""Pinned commit of the upstream repository (``develop`` on 2026-10-12)."""
MJCF_DIR = "src/mjlab_microduck/robot/microduck"
BALL_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Entertainment/Games/"
    "Ball_Walnut.usd"
)
"""Ball mesh of the NVIDIA ArchVis assets: one mesh in centimetres, 12.9 cm across, pivot at the bottom."""
CONTACT_OFFSET = 0.002
"""PhysX contact offset authored into ball and robot collider assets [m]."""
MODEL = "robot_walk"
"""Model to download: the MJCF is ``<MODEL>.xml``, the USD ``<MODEL>/<MODEL>.usda``."""

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "asset")


def find_isaaclab_root() -> str:
    """Return the first parent directory containing `scripts/tools/convert_mjcf.py`.

    Raise `RuntimeError` if the converter cannot be found.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    while path != os.path.dirname(path):
        if os.path.isfile(os.path.join(path, "scripts", "tools", "convert_mjcf.py")):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("Could not find the Isaac Lab checkout above this file.")


def fetch(name: str, destination: str, local: str | None) -> None:
    """Copies ``name`` (relative to the MJCF folder) from a local checkout, or downloads it from GitHub."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if local is not None:
        shutil.copyfile(os.path.join(local, name), destination)
        return
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{MJCF_DIR}/{name}"
    print(f"  downloading {name}", flush=True)
    urllib.request.urlretrieve(url, destination)


def build_ball(asset_dir: str, source: str | None) -> str:
    """Write `ball/ball.usda` under `asset_dir` and return its path.

    Scale the visual mesh to unit radius and add a separate sphere collider.
    Remove rigid-body, collision, and mesh-collision APIs from the referenced
    visual prims where present.

    When `source` is None, use the bundled basketball if available. Otherwise,
    use `Ball_Walnut.usd` and download it if the local file is missing.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    ball_dir = os.path.join(asset_dir, "ball")
    os.makedirs(ball_dir, exist_ok=True)
    if source is None:
        basketball = glob.glob(os.path.join(asset_dir, "basketball", "model_*.usd"))
        source = basketball[0] if basketball else os.path.join(ball_dir, "Ball_Walnut.usd")
    if source.endswith("Ball_Walnut.usd"):
        if not os.path.isfile(source):
            print("  downloading Ball_Walnut.usd", flush=True)
            urllib.request.urlretrieve(BALL_URL, source)
    source = os.path.abspath(source)
    # size and centre of the mesh, in its own units
    mesh_stage = Usd.Stage.Open(source)
    default_prim = mesh_stage.GetDefaultPrim().GetPath()
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(mesh_stage.GetPseudoRoot())
    box = box.ComputeAlignedBox()
    centre, radius = (box.GetMin() + box.GetMax()) / 2, max(box.GetSize()) / 2
    usd_file = os.path.join(ball_dir, "ball.usda")
    stage = Usd.Stage.CreateNew(usd_file)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/ball")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    body.CreateRigidBodyEnabledAttr().Set(True)
    body.CreateKinematicEnabledAttr().Set(False)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(0.3)
    visual = UsdGeom.Xform.Define(stage, "/ball/visual")
    visual.GetPrim().GetReferences().AddReference(os.path.relpath(source, ball_dir), default_prim)
    visual.AddTranslateOp().Set(Gf.Vec3d(-centre[0], -centre[1], -centre[2]) / radius)
    visual.AddScaleOp().Set(Gf.Vec3f(1.0 / radius))
    # the visual must not simulate: drop the rigid bodies and colliders that the source ball may bring
    for prim in Usd.PrimRange(visual.GetPrim()):
        for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI):
            if prim.HasAPI(schema):
                prim.RemoveAPI(schema)
    collider = UsdGeom.Sphere.Define(stage, "/ball/collider")
    collider.GetRadiusAttr().Set(1.0)
    collider.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    # PhysX makes a contact active within 2 cm of a shape unless told otherwise, which is too much next to a 5 cm foot
    collider.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    collider.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
    collider.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    stage.GetRootLayer().Save()
    centre = tuple(round(c, 4) for c in centre)
    print(f"[build_asset] ball {os.path.basename(source)}: radius {radius:.4f}, centre {centre}")
    return usd_file


def main():
    parser = argparse.ArgumentParser(description="Download Microduck and convert it to USD.")
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help=f"Folder holding {MODEL}.xml and its assets/ (e.g. .../{MJCF_DIR}) instead of downloading.",
    )
    parser.add_argument("--force", action="store_true", help="Download and convert again even if the USD exists.")
    parser.add_argument(
        "--ball",
        type=str,
        default=None,
        help="USD file of the ball (for example a SimReady basketball) instead of the downloaded walnut ball.",
    )
    parser.add_argument("--self_collision", action="store_true", help="Enable self-collisions between the links.")
    args, convert_args = parser.parse_known_args()

    asset_dir = os.path.abspath(ASSET_DIR)
    source_dir = os.path.join(asset_dir, "source")
    usd_file = os.path.join(asset_dir, MODEL, f"{MODEL}.usda")
    if os.path.isfile(usd_file) and not args.force:
        print(f"[build_asset] {usd_file} exists (use --force to rebuild).")
    else:
        # 1. the MJCF and the meshes it names
        if args.force:
            shutil.rmtree(source_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(asset_dir, MODEL), ignore_errors=True)
        print(f"[build_asset] fetching {MODEL}.xml from {args.local or f'github.com/{REPO}@{COMMIT[:7]}'}")
        mjcf = os.path.join(source_dir, f"{MODEL}.xml")
        fetch(f"{MODEL}.xml", mjcf, args.local)
        root = ET.parse(mjcf).getroot()
        mesh_dir = root.find("compiler").get("meshdir", ".")
        meshes = sorted({m.get("file") for m in root.iter("mesh")})
        for name in meshes:
            fetch(f"{mesh_dir}/{name}", os.path.join(source_dir, mesh_dir, name), args.local)
        print(f"[build_asset] {len(meshes)} meshes ready in {source_dir}")

        # 2. MJCF -> USD with Isaac Lab's converter
        command = [
            sys.executable,
            os.path.join(find_isaaclab_root(), "scripts", "tools", "convert_mjcf.py"),
            mjcf,
            os.path.join(asset_dir, f"{MODEL}.usda"),  # the importer writes <asset_dir>/<MODEL>/<MODEL>.usda
            *(["--self_collision"] if args.self_collision else []),
            *convert_args,
        ]
        print("[build_asset] converting:", " ".join(command[1:]), flush=True)
        subprocess.run(command, check=True)
        if not os.path.isfile(usd_file):
            raise RuntimeError(f"The converter did not write {usd_file}.")
    # Author offsets in collider definitions: runtime overrides cannot edit instance proxies.
    from pxr import Sdf, Usd, UsdPhysics

    instances = os.path.join(asset_dir, MODEL, "payloads", "instances.usda")
    if not os.path.isfile(instances):
        raise RuntimeError("Converted asset layout changed: inspect robot colliders before using this build.")
    collider_stage = Usd.Stage.Open(instances)
    colliders = [p for p in collider_stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise RuntimeError("No robot colliders found; contact offsets were not authored.")
    for prim in colliders:
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(CONTACT_OFFSET)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    collider_stage.GetRootLayer().Save()
    build_ball(asset_dir, args.ball)
    print(f"[build_asset] done: {usd_file}")


if __name__ == "__main__":
    main()
