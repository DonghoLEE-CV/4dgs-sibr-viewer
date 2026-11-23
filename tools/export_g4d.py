"""Convert a 4DGaussians checkpoint into a SIBR-4D model folder.

The deformation field (HexPlane + MLP) is evaluated with the original 4DGaussians code
(scene/deformation.py, cloned by setup.bat into third_party/4DGaussians) on every time step, and every
frame's deformed Gaussians are written to sequence.g4d so the viewer can play them back without PyTorch.

Output folder (same layout as a 3DGS model folder, so it also opens in the original SIBR viewer as a static model):
  cfg_args                                   copied from the checkpoint
  cameras.json                               training cameras (3DGS format) for camera snapping
  point_cloud/iteration_N/point_cloud.ply    canonical Gaussians (copied unchanged)
  sequence.g4d                               all frames (see format below)

sequence.g4d (little endian):
  header 32 B : char[4] 'G4D1' | int32 N | int32 F | int32 SH coeffs (16) | float32 t_last | 12 B padding
  per frame   : pos float32[N*3] | rot fp16[N*4] (normalized wxyz) | scale fp16[N*3] (exp) |
                opacity fp16[N] (sigmoid) | shs fp16[N*48] (coefficient-major, rgb inner)

usage (run through export_4dgs.bat, which uses the environment made by setup.bat):
  export_4dgs.bat --ckpt <4DGaussians output folder> --data <dataset scene folder> --out <new model folder>
    --ckpt  folder holding cfg_args and point_cloud/iteration_N/ (or iteration_N/ directly)
    --data  D-NeRF scene (has transforms_train.json) or Neu3D/DyNeRF scene (has poses_bounds.npy)
    --frames  number of time steps (default: 300 for Neu3D, 100 for D-NeRF)
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import types
from argparse import Namespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CODE = os.path.join(os.path.dirname(HERE), "third_party", "4DGaussians")


# ----------------------------------------------------------------------------- 4DGaussians model (no compiled extensions)
def load_repo_modules(code):
    """import scene.deformation without running scene/__init__.py (that one needs compiled CUDA extensions)"""
    for pkg in ("scene", "utils"):
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(code, pkg)]
        sys.modules[pkg] = m

    def load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(code, rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    load("utils.graphics_utils", "utils/graphics_utils.py")
    load("scene.hexplane", "scene/hexplane.py")
    load("scene.grid", "scene/grid.py")
    return load("scene.deformation", "scene/deformation.py")


def read_cfg(ckpt):
    return eval(open(os.path.join(ckpt, "cfg_args")).read(), {"Namespace": Namespace})


def latest_iter(ckpt):
    its = glob.glob(os.path.join(ckpt, "iteration_*")) + glob.glob(os.path.join(ckpt, "point_cloud", "iteration_*"))
    if not its:
        sys.exit(f"no iteration_* folder under {ckpt}")
    return max(its, key=lambda p: int(p.rsplit("_", 1)[1]))


def read_ply_vertices(path):
    """minimal reader for binary little-endian PLY files with float/double/uchar vertex properties (3DGS layout)"""
    types = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
             "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2", "ushort": "<u2"}
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            sys.exit(f"{path}: not a PLY file")
        n, props, in_vertex = 0, [], False
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format") and "binary_little_endian" not in line:
                sys.exit(f"{path}: only binary_little_endian PLY is supported")
            if line.startswith("element"):
                _, name, cnt = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    n = int(cnt)
            elif line.startswith("property") and in_vertex:
                _, t, name = line.split()
                props.append((name, types[t]))
            elif line == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=n)


def load_ply(path, dev):
    v = read_ply_vertices(path)
    names = list(v.dtype.names)
    g = lambda k: np.asarray(v[k], dtype=np.float32)
    xyz = np.stack([g("x"), g("y"), g("z")], 1)
    dc = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], 1)[:, None, :]
    rest_n = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    rest = np.stack([g(n) for n in rest_n], 1).reshape(len(xyz), 3, -1).transpose(0, 2, 1)
    sc = np.stack([g(n) for n in sorted([n for n in names if n.startswith("scale_")])], 1)
    rot = np.stack([g(n) for n in sorted([n for n in names if n.startswith("rot")])], 1)
    op = g("opacity")[:, None]
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    return dict(xyz=t(xyz), shs=torch.cat([t(dc), t(rest)], 1), scale=t(sc), rot=t(rot), opacity=t(op))


class Model4DGS:
    def __init__(self, ckpt, code, dev="cuda"):
        if not os.path.isfile(os.path.join(code, "scene", "deformation.py")):
            sys.exit(f"4DGaussians code not found at {code} (run setup.bat first, or pass --code)")
        D = load_repo_modules(code)
        self.cfg = read_cfg(ckpt)
        it = latest_iter(ckpt)
        self.iter_dir, self.iter_name = it, os.path.basename(it)
        self.g = load_ply(os.path.join(it, "point_cloud.ply"), dev)
        self.net = D.deform_network(self.cfg).to(dev).eval()
        self.net.load_state_dict(torch.load(os.path.join(it, "deformation.pth"), map_location=dev, weights_only=False))
        self.n = self.g["xyz"].shape[0]


# ----------------------------------------------------------------------------- cameras.json (3DGS format)
def cams_dnerf(data):
    d = json.load(open(os.path.join(data, "transforms_train.json")))
    W = H = 800
    f = 0.5 * W / math.tan(0.5 * d["camera_angle_x"])
    out = []
    for i, fr in enumerate(d["frames"]):
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1                      # OpenGL -> COLMAP axes, as in 3DGS readCamerasFromTransforms
        out.append(dict(id=i, img_name=os.path.basename(fr["file_path"]), width=W, height=H,
                        position=c2w[:3, 3].tolist(), rotation=c2w[:3, :3].tolist(), fx=f, fy=f))
    return out


def cams_dynerf(data):
    """same conversion as 4DGaussians scene/neural_3D_dataset_NDC.py (half resolution 1352 x 1014)"""
    arr = np.load(os.path.join(data, "poses_bounds.npy"))
    poses = arr[:, :-2].reshape([-1, 3, 5])
    W, H = 1352, 1014
    focal = poses[0, 2, 4] / (2704 / W)
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    out = []
    for i, p in enumerate(poses):
        R = -p[:3, :3]
        R[:, 0] = -R[:, 0]
        T = -p[:3, 3].dot(R)
        out.append(dict(id=i, img_name=f"cam{i:02d}", width=W, height=H,
                        position=(-R @ T).tolist(), rotation=R.tolist(), fx=float(focal), fy=float(focal)))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="", help="dataset scene folder (default: look in --ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--code", default=DEFAULT_CODE)
    a = ap.parse_args()

    data = a.data or a.ckpt
    if os.path.isfile(os.path.join(data, "transforms_train.json")):
        kind = "dnerf"
    elif os.path.isfile(os.path.join(data, "poses_bounds.npy")):
        kind = "dynerf"
    else:
        sys.exit(f"--data {data}: needs transforms_train.json (D-NeRF) or poses_bounds.npy (Neu3D / DyNeRF)")

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available to PyTorch")
    m = Model4DGS(a.ckpt, a.code)
    F = a.frames or (300 if kind == "dynerf" else 100)
    ts = [i / 300.0 for i in range(F)] if kind == "dynerf" else [i / max(F - 1, 1) for i in range(F)]

    it = os.path.join(a.out, "point_cloud", m.iter_name)
    os.makedirs(it, exist_ok=True)
    shutil.copyfile(os.path.join(m.iter_dir, "point_cloud.ply"), os.path.join(it, "point_cloud.ply"))
    shutil.copyfile(os.path.join(a.ckpt, "cfg_args"), os.path.join(a.out, "cfg_args"))
    cams = cams_dnerf(data) if kind == "dnerf" else cams_dynerf(data)
    json.dump(cams, open(os.path.join(a.out, "cameras.json"), "w"))

    g, N = m.g, m.n
    path = os.path.join(a.out, "sequence.g4d")
    with open(path, "wb") as fo:
        fo.write(b"G4D1" + struct.pack("<iiif", N, F, 16, ts[-1]) + b"\0" * 12)
        for k, t in enumerate(ts):
            tt = torch.full((N, 1), float(t), device="cuda")
            xyz, sc, rot, op, shs = m.net(g["xyz"], g["scale"], g["rot"], g["opacity"], g["shs"], tt)
            rot = torch.nn.functional.normalize(rot, dim=-1)
            fo.write(xyz.float().contiguous().cpu().numpy().tobytes())
            for x in (rot, torch.exp(sc), torch.sigmoid(op), shs.reshape(N, 48)):
                fo.write(x.half().contiguous().cpu().numpy().tobytes())
            if k % 50 == 0:
                print(f"frame {k}/{F}", flush=True)
    print(f"wrote {path}: {kind}, N={N}, frames={F}, {os.path.getsize(path) / 1e9:.2f} GB, cameras={len(cams)}")


if __name__ == "__main__":
    main()
