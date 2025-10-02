#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_orbit_rgbd.py — GLB/GLTF/OBJ/PLY batch-parallel RGBD rendering with multiple camera rings (PyTorch3D)

Features
- Load .glb/.gltf/.obj/.ply via trimesh → PyTorch3D Meshes
- Vertex color or UV baseColor texture (optional), simple Phong lighting (non-PBR)
- Metric depth via unproject (camera-space Z, in meters)
- Equatorial ring (azimuth sweep), Top ring (high-elevation sweep), Latitudinal ring (elevation sweep at a fixed azimuth)
- Rings are concatenated in order into one video
- Orientation control: apply glTF scene transforms; optional axis correction (e.g., Y-up → Z-up)
- Options for backface culling, background color, batching to avoid OOM

Usage examples
    # Base equatorial ring only
    python render_orbit_rgbd.py \
      --obj ../Data/aligned_mugs/mug_1532_7659420.glb \
      --num_cams 80 --elev_deg 20 --image_size 512 --fov_deg 60 \
      --seed 123 --save_metric_depth --make_video \
      --out ../Data/out_orbit

    # Add a top (overhead) ring, bring camera closer for inside-cup view, keep one video
    python render_orbit_rgbd.py \
      --obj ../Data/aligned_mugs/mug_1532_7659420.glb \
      --num_cams 80 --elev_deg 20 \
      --top_ring_num 40 --top_ring_elev_deg 75 --top_ring_dist_scale 0.85 \
      --image_size 512 --fov_deg 60 --seed 123 \
      --use_uv_textures --no_cull_backfaces \
      --save_metric_depth --make_video --out ../Data/out_orbit

    # Add a latitudinal ring (sweep elevation at fixed azimuth, i.e., around a horizontal axis)
    python render_orbit_rgbd.py \
      --obj ../Data/aligned_mugs/mug_1532_7659420.glb \
      --num_cams 80 --elev_deg 20 \
      --lat_ring_num 40 --lat_ring_azim_deg 0 --lat_ring_min_elev_deg 5 --lat_ring_max_elev_deg 175 \
      --image_size 512 --fov_deg 60 --seed 123 \
      --use_uv_textures --no_cull_backfaces \
      --save_metric_depth --make_video --out ../Data/out_orbit

Dependencies
    pip install torch torchvision pytorch3d trimesh imageio imageio-ffmpeg

Notes
- This script prefers vertex colors. If --use_uv_textures is set and a baseColor texture is present, TexturesUV is used.
- True PBR (metallic/roughness/normal) is NOT implemented here. Can be added on request.
- Intrinsics are derived from vertical FoV and image size; depth is camera-space Z (meters), invalid pixels set to 0.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

import imageio
import numpy as np
import torch
import h5py
# --- PyTorch3D imports (with version fallbacks) ---
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    HardPhongShader,
    PointLights,
)
try:
    # new location
    from pytorch3d.renderer import BlendParams
except Exception:  # pragma: no cover
    # older versions
    from pytorch3d.renderer.blending import BlendParams  # type: ignore
from pytorch3d.renderer.mesh import TexturesVertex
# --- TexturesUV import across PyTorch3D versions ---
try:
    from pytorch3d.renderer import TexturesUV as _TexturesUV
except Exception:  # older versions
    try:
        from pytorch3d.renderer.mesh import TexturesUV as _TexturesUV
    except Exception:
        _TexturesUV = None  # UV textures not available; we'll gracefully fall back

# Alias for usage
TexturesUV = _TexturesUV
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.renderer import SoftPhongShader, Materials, BlendParams
# trimesh for loading glb/gltf/obj/ply
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesAtlas  # 用于老接口打图集

import warnings
warnings.filterwarnings(
    "ignore", message="All-NaN slice encountered",
    category=RuntimeWarning, module="trimesh.visual.texture"
)
warnings.filterwarnings(
    "ignore", message="invalid value encountered in cast",
    category=RuntimeWarning, module="trimesh.visual.texture"
)

# --- PyTorch3D OBJ loader capability probe (robust; avoid importing pytorch3d.io __init__) ---
HAVE_P3D = False
HAVE_LOAD_OBJS_AS_MESHES = False
HAVE_LOAD_OBJ = False
_P3D_IMPORT_ERR = None
import os
if os.getenv("DEBUG_ATTACH", "0") == "1":
    import debugpy
    debugpy.listen(("127.0.0.1", int(os.getenv("DEBUG_CHILD_PORT", "5679"))))
    print("[debug] waiting for VSCode attach...", flush=True)
    debugpy.wait_for_client()
    # debugpy.breakpoint()
try:
    import torch
    import importlib
    import pytorch3d  # 仅导根包
    HAVE_P3D = True
    try:
        # 直接导子模块，不触发 pytorch3d.io/__init__.py
        obj_io = importlib.import_module("pytorch3d.io.obj_io")
        if hasattr(obj_io, "load_objs_as_meshes"):
            from pytorch3d.io.obj_io import load_objs_as_meshes as _LOAD_OBJS_AS_MESHES
            HAVE_LOAD_OBJS_AS_MESHES = True
        if hasattr(obj_io, "load_obj"):
            from pytorch3d.io.obj_io import load_obj as _LOAD_OBJ
            HAVE_LOAD_OBJ = True
        if not (HAVE_LOAD_OBJS_AS_MESHES or HAVE_LOAD_OBJ):
            _P3D_IMPORT_ERR = "pytorch3d.io.obj_io has neither 'load_objs_as_meshes' nor 'load_obj'"
    except Exception as e:
        _P3D_IMPORT_ERR = e
except Exception as e:
    _P3D_IMPORT_ERR = e

from pytorch3d.structures import Meshes
# 放在文件顶部其它 import 附近
try:
    from pytorch3d.structures.utils import join_meshes_as_batch
except Exception:
    join_meshes_as_batch = None

def debug_print_p3d_capability_once():
    if not getattr(debug_print_p3d_capability_once, "_did", False):
        try:
            import pytorch3d, torch
            print(f"[p3d] torch={torch.__version__} cuda={getattr(torch.version,'cuda',None)}; "
                  f"pytorch3d={getattr(pytorch3d,'__version__','n/a')} "
                  f"HAVE_P3D={HAVE_P3D} HAVE_LOAD_OBJS_AS_MESHES={HAVE_LOAD_OBJS_AS_MESHES} HAVE_LOAD_OBJ={HAVE_LOAD_OBJ}")
            if _P3D_IMPORT_ERR:
                print(f"[p3d] detail: {_P3D_IMPORT_ERR}")
        except Exception:
            pass
        debug_print_p3d_capability_once._did = True

# ------------------------------ Utilities ------------------------------

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def axis_correction_matrix(kind: str, device: torch.device) -> torch.Tensor:
    """Return a 3x3 rotation to map common up-axis conventions.
    kind ∈ {"none", "y_up_to_z_up", "z_up_to_y_up"}.
    """
    if kind == "none":
        return torch.eye(3, device=device, dtype=torch.float32)
    if kind == "y_up_to_z_up":
        # rotate about +X by -90°: (x, y, z) → (x, z, -y)
        return torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device)
    if kind == "z_up_to_y_up":
        # rotate about +X by +90°: (x, y, z) → (x, -z, y)
        return torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device=device)
    raise ValueError(f"Unknown axis_correction: {kind}")
    
def _downscale_tex_np(tex_np: np.ndarray, max_edge: int) -> np.ndarray:
    """把纹理最长边限制到 max_edge；返回 float32 [0,1]。max_edge<=0 不缩放。"""
    if max_edge is None or int(max_edge) <= 0:
        if tex_np.dtype == np.uint8:
            return (tex_np.astype(np.float32) / 255.0)
        return tex_np.astype(np.float32, copy=False)
    H, W = tex_np.shape[:2]
    s = max(H, W)
    if s <= max_edge:
        if tex_np.dtype == np.uint8:
            return (tex_np.astype(np.float32) / 255.0)
        return tex_np.astype(np.float32, copy=False)
    # 首选 PIL 的高质量缩放
    try:
        from PIL import Image
        arr8 = tex_np
        if arr8.dtype != np.uint8:
            arr8 = np.clip(arr8, 0, 1)
            arr8 = (arr8 * 255).astype(np.uint8)
        im = Image.fromarray(arr8)
        newW = int(round(W * max_edge / s))
        newH = int(round(H * max_edge / s))
        im = im.resize((newW, newH), Image.LANCZOS)
        out = np.asarray(im)
        return out.astype(np.float32) / 255.0
    except Exception:
        # 兜底：线性下采样
        scale = max_edge / float(s)
        newH = max(1, int(round(H * scale)))
        newW = max(1, int(round(W * scale)))
        ys = (np.linspace(0, H-1, newH)).astype(np.int64)
        xs = (np.linspace(0, W-1, newW)).astype(np.int64)
        out = tex_np[ys][:, xs]
        if out.dtype == np.uint8:
            return out.astype(np.float32) / 255.0
        return out.astype(np.float32)
# ===== MTL parsing & helpers =====
def _srgb_to_linear_(maps: List[torch.Tensor]):
    # 原地把 [N,H,W,3] 或 [H,W,3] 的贴图做 sRGB→Linear
    def _convert(x: torch.Tensor):
        is_4d = (x.ndim == 4)
        img = x if not is_4d else x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        # piecewise gamma
        a = 0.04045
        low  = img <= a
        high = ~low
        img[low]  = img[low] / 12.92
        img[high] = ((img[high] + 0.055) / 1.055) ** 2.4
        return x
    for i in range(len(maps)):
        maps[i] = _convert(maps[i].clamp(0,1))
    return maps

def _parse_mtl_basic(mtl_path: Path) -> dict:
    """读出 Ka/Kd/Ks/Ns/Tr 或 d，以及 map_Kd 的 -s/-o/clamp on/off。"""
    out = {
        "Ka": (0.2, 0.2, 0.2),
        "Kd": (0.8, 0.8, 0.8),
        "Ks": (0.0, 0.0, 0.0),
        "Ns": 32.0,                 # Phong shininess
        "d": 1.0,                   # 透明度（1=不透明）
        "map_Kd_scale": (1.0, 1.0), # -s sx sy
        "map_Kd_offset": (0.0, 0.0),# -o ox oy
        "map_Kd_clamp": False,      # clamp on/off
    }
    try:
        txt = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return out
    clamp_flag = None
    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lo = s.lower()
        try:
            if lo.startswith("ka "):
                vals = [float(x) for x in s.split()[1:4]]
                if len(vals) >= 3: out["Ka"] = tuple(vals[:3])
            elif lo.startswith("kd "):
                vals = [float(x) for x in s.split()[1:4]]
                if len(vals) >= 3: out["Kd"] = tuple(vals[:3])
            elif lo.startswith("ks "):
                vals = [float(x) for x in s.split()[1:4]]
                if len(vals) >= 3: out["Ks"] = tuple(vals[:3])
            elif lo.startswith("ns "):
                out["Ns"] = float(s.split()[1])
            elif lo.startswith("tr "):
                out["d"] = 1.0 - float(s.split()[1])
            elif lo.startswith("d "):
                out["d"] = float(s.split()[1])
            elif lo.startswith("map_kd"):
                tokens = s.split()[1:]  # 选项 + 文件名
                i = 0
                while i < len(tokens):
                    t = tokens[i].lower()
                    if t == "-s" and i + 2 < len(tokens):
                        sx, sy = float(tokens[i+1]), float(tokens[i+2])
                        out["map_Kd_scale"] = (sx, sy); i += 3; continue
                    if t == "-o" and i + 2 < len(tokens):
                        ox, oy = float(tokens[i+1]), float(tokens[i+2])
                        out["map_Kd_offset"] = (ox, oy); i += 3; continue
                    if t == "-clamp" and i + 1 < len(tokens):
                        clamp_flag = tokens[i+1].lower() in ("on","1","true")
                        i += 2; continue
                    i += 1
        except Exception:
            continue
    if clamp_flag is not None:
        out["map_Kd_clamp"] = bool(clamp_flag)
    # 把 Ns 压到 PyTorch3D 合理区间（0~128 常见）
    out["Ns"] = max(1.0, min(128.0, out["Ns"]))
    # 颜色裁剪
    out["Ka"] = tuple(max(0.0, min(1.0, x)) for x in out["Ka"])
    out["Kd"] = tuple(max(0.0, min(1.0, x)) for x in out["Kd"])
    out["Ks"] = tuple(max(0.0, min(1.0, x)) for x in out["Ks"])
    return out

def _find_mtl_files_for_obj(obj_path: Path) -> List[Path]:
    mtls = []
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("mtllib"):
                    parts = line.split()[1:]
                    for p in parts:
                        mp = (obj_path.parent / p).resolve()
                        if mp.exists():
                            mtls.append(mp)
    except Exception:
        pass
    return mtls

def _apply_uv_transform_from_mtl(mesh: Meshes, offset_xy, scale_xy, clamp=False) -> Meshes:
    """按 MTL 的 -o/-s 变换 UV；支持 clamp/repeat。"""
    if not hasattr(mesh, "textures") or not isinstance(mesh.textures, TexturesUV):
        return mesh
    verts_uvs_list = mesh.textures.verts_uvs_list()
    faces_uvs_list = mesh.textures.faces_uvs_list()
    maps_list      = mesh.textures.maps_list()
    ox, oy = float(offset_xy[0]), float(offset_xy[1])
    sx, sy = float(scale_xy[0]),  float(scale_xy[1])

    new_uvs = []
    for uvs in verts_uvs_list:
        uv = uvs.clone()
        uv[:, 0] = uv[:, 0] * sx + ox
        uv[:, 1] = uv[:, 1] * sy + oy
        if clamp:
            uv = uv.clamp(0.0, 1.0)
        else:
            uv = uv.remainder(1.0)  # repeat
        new_uvs.append(uv)
    # 重新打包 TexturesUV
    new_tex = TexturesUV(maps=maps_list, faces_uvs=faces_uvs_list, verts_uvs=new_uvs)
    return Meshes(verts=mesh.verts_list(), faces=mesh.faces_list(), textures=new_tex)

def concat_trimesh_list(mesh_list: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(mesh_list) == 1:
        return mesh_list[0]
    return trimesh.util.concatenate(mesh_list)

def _estimate_face_count_from_obj(obj_path: Path, max_scan: int = 5_000_000) -> int:
    """快速估算 OBJ 面数（扫 f 行），避免完整加载前就 OOM。"""
    obj_path = Path(obj_path)
    cnt = 0
    with obj_path.open('r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if line.startswith('f ') or line.startswith('f\t'):
                cnt += 1
            if i >= max_scan:
                break
    return max(1, cnt)

def _atlas_bytes_estimate(faces: int, tile: int) -> int:
    # F × T × T × 3通道 × 4字节
    return int(faces) * int(tile) * int(tile) * 3 * 4

from pytorch3d.renderer import TexturesAtlas

def has_textures(mesh) -> bool:
    """
    仅做“存在性与类型”判断，避免访问会触发大额内存的函数。
    - TexturesAtlas：只要存在实例就认为“有贴图”（atlas 是 per-face 贴图，采样时即可使用）
    - TexturesUV：检查 maps_padded 的维度
    - TexturesVertex：检查 verts_features_padded 的维度
    """
    tx = getattr(mesh, "textures", None)
    if tx is None:
        return False

    # Atlas：存在实例即可认为有贴图（不调用 atlas_padded()，避免显存/内存开销）
    if isinstance(tx, TexturesAtlas):
        return True

    # UV：尽量轻量地确认 shape（某些版本 maps_padded 懒构建，读取一次足够）
    if isinstance(tx, TexturesUV):
        try:
            mp = tx.maps_padded()
            return (mp is not None) and (mp.ndim >= 4) and (mp.shape[-1] in (3, 4))
        except Exception:
            # 某些情况访问失败，但类型就是 UV，保守认为存在
            return True

    # Vertex：顶点颜色也算“有贴图”，Phong/SoftPhong 可正常采样
    if isinstance(tx, TexturesVertex):
        try:
            vf = tx.verts_features_padded()
            return (vf is not None) and (vf.ndim == 3) and (vf.shape[-1] == 3)
        except Exception:
            return True

    # 其它未知 textures 类型：保守认为存在，交给 shader 处理；不要提前降级
    return True

def attach_white_texture_atlas(mesh: Meshes, device: torch.device) -> Meshes:
    """
    给没有 textures 的 Meshes 附一个 1×1 的白色 per-face atlas，防止 SoftPhongShader 报错。
    """
    atlases = []
    for F_idx in mesh.faces_list():
        Fi = int(F_idx.shape[0])
        atlases.append(torch.ones((Fi, 1, 1, 3), dtype=torch.float32, device=device))
    textures = TexturesAtlas(atlas=atlases)
    return Meshes(verts=mesh.verts_list(), faces=mesh.faces_list(), textures=textures)

# === keep-colors helper: prefer vertex/face colors when no UV texture ===
import numpy as _np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

def _tm_to_p3d_with_colors(tri, device, albedo_rgb=(0.7, 0.7, 0.7)) -> Meshes:
    """Convert a trimesh.Trimesh to PyTorch3D Meshes with TexturesVertex.
       - Prefer vertex colors; if only face colors exist, average them to vertices.
       - Fall back to a flat albedo when no color is available.
    """
    if not isinstance(tri, trimesh.Trimesh) or tri.faces is None:
        raise TypeError("expect trimesh.Trimesh with faces")

    verts = torch.as_tensor(_np.asarray(tri.vertices), dtype=torch.float32, device=device)
    faces = torch.as_tensor(_np.asarray(tri.faces), dtype=torch.int64, device=device)
    V = verts.shape[0]

    vcol = None
    vis = getattr(tri, "visual", None)
    if vis is not None:
        # 1) vertex colors
        if getattr(vis, "vertex_colors", None) is not None:
            vc = _np.asarray(vis.vertex_colors)[..., :3].astype(_np.float32)
            if vc.max() > 1.0: vc = vc / 255.0
            if vc.shape[0] == V:
                vcol = torch.from_numpy(vc).to(device)
        # 2) face colors → average to vertices
        elif getattr(vis, "face_colors", None) is not None:
            fc = _np.asarray(vis.face_colors)[..., :3].astype(_np.float32)
            if fc.max() > 1.0: fc = fc / 255.0
            F = faces.shape[0]
            if fc.shape[0] == F:
                vcol = torch.zeros((V, 3), dtype=torch.float32, device=device)
                cnt  = torch.zeros((V, 1), dtype=torch.float32, device=device)
                idx  = faces.reshape(-1)                         # (F*3,)
                col  = torch.from_numpy(fc).to(device)           # (F,3)
                col  = col[:, None, :].expand(F, 3, 3).reshape(-1, 3)
                one  = torch.ones((idx.numel(), 1), dtype=torch.float32, device=device)
                vcol.index_add_(0, idx, col)
                cnt.index_add_(0, idx, one)
                mask = cnt.squeeze(-1) > 0
                vcol[mask] = vcol[mask] / cnt[mask]
                if (~mask).any():
                    vcol[~mask] = torch.tensor(albedo_rgb, dtype=torch.float32, device=device)

    if vcol is None:
        vcol = torch.tensor(albedo_rgb, dtype=torch.float32, device=device).expand(V, 3)

    tex = TexturesVertex(verts_features=[vcol])
    return Meshes(verts=[verts], faces=[faces], textures=tex)

# === NEW: enumerate directory sources by format (non-recursive) ===
from pathlib import Path
from typing import List, Tuple

def enumerate_targets(root: Path, input_format: str) -> List[Tuple[Path, str]]:
    """Return [(mesh_path, out_name)] collected from ROOT (non-recursive).
       - 'glb':  *.glb/*.gltf under root
       - 'obj':  *_obj folders (prefer <folder>/model.obj, else first *.obj), plus loose *.obj under root
       - 'auto': GLBs then OBJs (both of above)
    """
    targets: List[Tuple[Path, str]] = []

    def _collect_glbs():
        for f in sorted(root.glob('*.glb')):  targets.append((f, f.stem))
        for f in sorted(root.glob('*.gltf')): targets.append((f, f.stem))

    def _collect_objs():
        # *_obj folders at root
        for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name.endswith('_obj')],
                        key=lambda p: (len(p.parts), str(p))):
            of = d / 'model.obj'
            if not of.exists():
                cands = sorted(d.glob('*.obj'))
                if not cands:
                    continue
                of = cands[0]
            targets.append((of, d.name))
        # Loose OBJ files under root
        for f in sorted(root.glob('*.obj')):
            targets.append((f, f.stem))

    if input_format == 'glb':
        _collect_glbs()
    elif input_format == 'obj':
        _collect_objs()
    else:  # auto
        _collect_glbs()
        _collect_objs()

    # Dedup names (rare but safe): append _1, _2, ...
    seen = {}
    deduped: List[Tuple[Path, str]] = []
    for p, name in targets:
        n = name
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[name]}"
        else:
            seen[n] = 0
        deduped.append((p, n))
    return deduped

def mtl_has_maps(obj_path: Path) -> bool:
    mtls = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("mtllib"):
                mtls += [Path(obj_path.parent, p) for p in line.split()[1:]]
    for m in mtls:
        if not m.exists(): 
            continue
        txt = m.read_text(encoding="utf-8", errors="ignore").lower()
        if "map_" in txt:    # 很宽松，但足够判断“是否存在贴图指令”
            return True
    return False

def _n_meshes(meshes: Meshes) -> int:
    """robust: PyTorch3D 各版本都能拿到批大小"""
    try:
        return len(meshes)                  # 通用
    except Exception:
        return getattr(meshes, "_N", 1)     # 兜底

def _extend_meshes(meshes: Meshes, k: int) -> Meshes:
    """把单个 mesh 扩成 k 份；优先用 extend，缺失时用 join_meshes_as_batch 回退"""
    if k <= 1:
        return meshes
    if hasattr(meshes, "extend"):
        return meshes.extend(k)
    if join_meshes_as_batch is not None:
        # 注意：一些版本的 join_meshes_as_batch 需要列表输入
        return join_meshes_as_batch([meshes for _ in range(k)])
    # 最后兜底：退回单 mesh（会在后面被逻辑拦住）
    return meshes
# ------------------------------ Loading ------------------------------

def load_obj_with_textures_p3d(
    obj_path: str | Path,
    device: torch.device,
    *,
    use_atlas: bool = False,             # ★ 默认 False：使用 UV 贴图
    atlas_size: int = 8,
    atlas_mem_limit_gb: float = 8.0,     # ★ 超限自动关掉 atlas
    axis_correction: str = "none",
) -> Meshes:
    """
    优先使用 PyTorch3D 新接口 load_objs_as_meshes：
      - 默认 create_texture_atlas=False（使用 UV 贴图，省内存）
      - 若 use_atlas=True，则先估算 F 和内存，超限自动改回 UV
    """
    debug_print_p3d_capability_once()
    if not (HAVE_P3D and HAVE_LOAD_OBJS_AS_MESHES):
        raise RuntimeError("pytorch3d load_objs_as_meshes not available in this environment.")

    obj_path = Path(obj_path)
    from importlib import import_module
    _io = import_module("pytorch3d.io")
    # 若请求 atlas，先做熔断检查
    create_atlas = bool(use_atlas)
    if create_atlas:
        F = _estimate_face_count_from_obj(obj_path)
        bytes_est = _atlas_bytes_estimate(F, atlas_size)
        limit = int(atlas_mem_limit_gb * (1024**3))
        if bytes_est > limit:
            print(f"[warn] atlas memory estimate {bytes_est/1e9:.1f} GB > limit {atlas_mem_limit_gb} GB; "
                  f"auto-switch to UV textures (no atlas).  (faces≈{F}, tile={atlas_size})")
            create_atlas = False

    # ---- 1) 先试 UV 贴图模式 ----
    def _load(uv_mode: bool):
        return _io.load_objs_as_meshes(
            [str(obj_path)],
            device=device,
            load_textures=True,
            create_texture_atlas=not uv_mode,
            texture_atlas_size=int(atlas_size),
            texture_wrap="repeat",
            path_manager=None,  # 相对路径按 obj 所在目录解析
        )

    def _has_valid_textures(mesh: Meshes) -> bool:
        # TexturesUV 或 TexturesAtlas 都可以，关键是非空
        tex = getattr(mesh, "textures", None)
        if tex is None:
            return False
        # TexturesUV: maps() 非空；TexturesAtlas: atlas_packed() 非空
        try:
            if hasattr(tex, "maps") and len(tex.maps_list()) > 0:
                m = tex.maps_list()[0]
                return (m is not None) and (m.numel() > 0)
        except Exception:
            pass
        try:
            if hasattr(tex, "atlas_packed") and tex.atlas_packed() is not None:
                a = tex.atlas_packed()
                return (a is not None) and (a.numel() > 0)
        except Exception:
            pass
        return False

    # 先试 UV ...
    mesh = _load(uv_mode=True)
    if not _has_valid_textures(mesh):
        print("[warn] UV texture load failed or empty; ", end="")
        if create_atlas and mtl_has_maps(obj_path):
            print(f"retry with atlas (tile={atlas_size})...")
            mesh = _load(uv_mode=False)
        else:
            print("skip atlas (no map_* or mem-limit); falling back to trimesh loader.")
            # try:
            #     # 用 trimesh 读取，不做繁重处理
            #     tri_or_scene = trimesh.load(str(obj_path), force="mesh", process=False)
            #     if isinstance(tri_or_scene, trimesh.Scene):
            #         # 合并到一个 Trimesh（与 load_mesh_any 的 concat 逻辑一致）
            #         geoms = [g.as_trimesh() if hasattr(g, "as_trimesh") else g
            #                  for g in tri_or_scene.geometry.values()
            #                  if isinstance(g, (trimesh.Trimesh,))]
            #         tri = concat_trimesh_list([g.copy() for g in geoms if isinstance(g, trimesh.Trimesh)])
            #     else:
            #         tri = tri_or_scene
            #     mesh = _tm_to_p3d_with_colors(tri, device, albedo_rgb=(0.7, 0.7, 0.7))
            #     # 轴系修正（与下方一致）
            #     Rfix = axis_correction_matrix(axis_correction, device)
            #     if not torch.allclose(Rfix, torch.eye(3, device=device)):
            #         Vs = [V @ Rfix.T for V in mesh.verts_list()]
            #         Fs = mesh.faces_list()
            #         mesh = Meshes(verts=Vs, faces=Fs, textures=mesh.textures)
            #     print("[keep-colors] OBJ has no usable UV; using vertex/face colors → TexturesVertex")
            #     return mesh
            # except Exception as _e_keep:
            #     print(f"[warn] keep-colors path failed ({_e_keep}); will raise UV-missing error.")
            #     raise RuntimeError("UV textures missing; atlas disabled")
            raise RuntimeError("UV textures missing; atlas disabled")

            # if not _has_valid_textures(mesh):
            #     # 到这还不行，给出明确诊断，方便你定位 MTL/路径问题
            #     raise RuntimeError(
            #         "PyTorch3D failed to attach textures for this OBJ.\n"
            #         "可能原因：\n"
            #         "  - OBJ 未正确引用 .mtl（缺少 mtllib/usemtl）；\n"
            #         "  - .mtl 未包含 map_Kd / 贴图路径错误；\n"
            #         "  - 贴图路径为相对路径，但当前工作目录或解析目录不对；\n"
            #         "  - 贴图格式 PyTorch3D 无法解码；\n"
            #         "建议：用低层 _io.load_obj() 检查 aux.texture_images 是否非空；"
            #         "或临时导出为 atlas（create_texture_atlas=True）以规避多材质拼接问题。"
            #     )

    # ---- 2) 轴系修正：只旋转顶点，不触碰 textures ----
    Rfix = axis_correction_matrix(axis_correction, device)
    if not torch.allclose(Rfix, torch.eye(3, device=device)):
        verts_fixed = [V @ Rfix.T for V in mesh.verts_list()]
        faces_list  = mesh.faces_list()
        # 确保 textures 也在目标 device（有些版本 textures 默认在 CPU）
        tex = mesh.textures
        try:
            tex = tex.to(device)
        except Exception:
            pass
        mesh = Meshes(verts=verts_fixed, faces=faces_list, textures=tex)

    return mesh

def load_mesh_any(
    path: str | Path,
    device: torch.device,
    *,
    albedo: float = 0.7,
    use_uv_textures: bool = False,
    flip_v: bool = False,
    apply_scene_xform: bool = True,
    axis_correction: str = "none",
) -> Meshes:
    """Load GLB/GLTF/OBJ/PLY into a PyTorch3D Meshes object.

    - Applies scene graph transforms when requested.
    - Optional axis correction (e.g., y-up → z-up) applied after baking transforms.
    - Uses UV baseColor texture when available and --use_uv_textures is set; otherwise falls back to vertex colors or a flat albedo.
    """
    path = str(path)
    obj = trimesh.load(path, force="scene", process=False, maintain_order=True)
    if isinstance(obj, trimesh.Trimesh):
        scene_or_mesh = trimesh.Scene(geometry=[obj])  # 包一层，复用同一逻辑
    else:
        scene_or_mesh = obj
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError(f"No geometry in scene: {path}")
        # Robust handling of scene graph across trimesh versions
        def _as_tri(g):
            return g.as_trimesh() if hasattr(g, "as_trimesh") else g
        if apply_scene_xform:
            meshes: List[trimesh.Trimesh] = []
            try:
                for ng in scene_or_mesh.graph.nodes_geometry:
                    # nodes_geometry may yield (node_name, geom_name) or more; take first two
                    if isinstance(ng, (list, tuple)) and len(ng) >= 2:
                        node_name, geom_name = ng[0], ng[1]
                    else:
                        continue
                    if geom_name is None or geom_name not in scene_or_mesh.geometry:
                        continue
                    g0 = _as_tri(scene_or_mesh.geometry[geom_name])
                    if not isinstance(g0, trimesh.Trimesh):
                        continue
                    g = g0.copy()
                    get_res = scene_or_mesh.graph.get(node_name)
                    T = get_res[0] if isinstance(get_res, (list, tuple)) else get_res
                    if T is None:
                        import numpy as _np
                        T = _np.eye(4)
                    g.apply_transform(T)
                    meshes.append(g)
                if len(meshes) == 0:
                    # Fallback: no nodes (e.g., cameras-only); take raw geometries without transforms
                    meshes = []
                    for g in scene_or_mesh.geometry.values():
                        g = _as_tri(g)
                        if isinstance(g, trimesh.Trimesh):
                            meshes.append(g.copy())
            except Exception as e:
                print(f"[warn] apply_scene_xform failed ({e}); falling back without transforms.")
                meshes = []
                for g in scene_or_mesh.geometry.values():
                    g = _as_tri(g)
                    if isinstance(g, trimesh.Trimesh):
                        meshes.append(g.copy())
            tri = concat_trimesh_list(meshes)
        else:
            tri = concat_trimesh_list([
                (_as_tri(g).copy() if isinstance(_as_tri(g), trimesh.Trimesh) else _as_tri(g))
                for g in scene_or_mesh.geometry.values()
                if isinstance(_as_tri(g), trimesh.Trimesh)
            ])
    else:
        tri = scene_or_mesh

    if not isinstance(tri, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not Trimesh: {type(tri)} from {path}")

    verts = to_tensor(tri.vertices, device).contiguous()
    faces = to_tensor(tri.faces, device, dtype=torch.int64).contiguous()

    # Axis correction last, so it's in your canonical coordinates
    Rfix = axis_correction_matrix(axis_correction, device)
    if not torch.allclose(Rfix, torch.eye(3, device=device)):
        verts = verts @ Rfix.T

    # --- Try UV texture ---
    if use_uv_textures and (TexturesUV is not None) and hasattr(tri.visual, "uv") and tri.visual.uv is not None:
        uv = np.asarray(tri.visual.uv, dtype=np.float32)
        if flip_v:
            uv[:, 1] = 1.0 - uv[:, 1]

        tex_img = None
        if hasattr(tri.visual, "material") and getattr(tri.visual.material, "image", None) is not None:
            img = tri.visual.material.image
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img] * 3, -1)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # tex_img = torch.as_tensor(img, dtype=torch.float32, device=device) / 255.0
            # img 可能是 PIL.Image 或 只读的 numpy 视图：强制复制成可写的连续数组
            arr = np.array(img, copy=True)            # HxWxC, uint8
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8, copy=False)
            arr = arr.astype(np.float32, copy=False) / 255.0  # 转 float32 且归一化
            tex_img = torch.from_numpy(arr).to(device)        # 不用 as_tensor，避免把只读视图挂进来

        if tex_img is not None:
            verts_uvs = torch.as_tensor(uv, dtype=torch.float32, device=device)
            # In many trimesh cases, face indices can be reused for faces_uvs when UVs are vertex-aligned
            faces_uvs = faces.clone()
            textures = TexturesUV(
                maps=tex_img[None, ...],  # (1,H,W,3)
                faces_uvs=[faces_uvs],
                verts_uvs=[verts_uvs],
            )
            return Meshes(verts=[verts], faces=[faces], textures=textures)

    # If user asked for UV textures but TexturesUV is unavailable, warn once and fall back
    if use_uv_textures and TexturesUV is None:
        print("[warn] --use_uv_textures requested but TexturesUV is unavailable in your PyTorch3D; falling back to vertex colors / albedo.")

        # --- Fallback to vertex/face colors or flat albedo ---
    verts_rgb = None
    if hasattr(tri, "visual") and getattr(tri.visual, "vertex_colors", None) is not None:
        vc = np.asarray(tri.visual.vertex_colors)[..., :3].astype(np.float32)
        if vc.max() > 1.0: vc = vc / 255.0
        if vc.shape[0] == verts.shape[0]:
            verts_rgb = torch.from_numpy(vc).to(device)
    elif hasattr(tri, "visual") and getattr(tri.visual, "face_colors", None) is not None:
        fc = np.asarray(tri.visual.face_colors)[..., :3].astype(np.float32)
        if fc.max() > 1.0: fc = fc / 255.0
        V = verts.shape[0]; F = faces.shape[0]
        if fc.shape[0] == F:
            verts_rgb = torch.zeros((V, 3), dtype=torch.float32, device=device)
            cnt = torch.zeros((V, 1), dtype=torch.float32, device=device)
            idx = faces.reshape(-1)
            col = torch.from_numpy(fc).to(device)
            col = col[:, None, :].expand(F, 3, 3).reshape(-1, 3)
            one = torch.ones((idx.numel(), 1), dtype=torch.float32, device=device)
            verts_rgb.index_add_(0, idx, col)
            cnt.index_add_(0, idx, one)
            mask = cnt.squeeze(-1) > 0
            verts_rgb[mask] = verts_rgb[mask] / cnt[mask]

    if verts_rgb is None:
        verts_rgb = torch.full((verts.shape[0], 3), float(albedo), device=device)

    textures = TexturesVertex(verts_features=[verts_rgb])
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def load_mesh_any_glb_uv(path, device, albedo: float = 0.7,
    use_uv_textures: bool = False,
    flip_v: bool = False,
    apply_scene_xform: bool = True,
    axis_correction: str = "none"):
    scene = trimesh.load(path, force='scene')
    assert isinstance(scene, trimesh.Scene), "GLB应读成Scene"
    
    verts_list, faces_list = [], []
    verts_uvs_list, faces_uvs_list, maps_list = [], [], []

    for name, geom in scene.geometry.items():
        tri = geom.as_trimesh() if hasattr(geom, "as_trimesh") else geom
        if not isinstance(tri, trimesh.Trimesh) or tri.faces is None:
            continue

        # 顶点/面
        V = torch.tensor(np.asarray(tri.vertices), dtype=torch.float32, device=device)
        F = torch.tensor(np.asarray(tri.faces), dtype=torch.int64, device=device)

        # --- 读取UV + baseColor纹理 ---
        uv = None
        tex_img = None
        if isinstance(tri.visual, trimesh.visual.texture.TextureVisuals):
            uv = np.asarray(tri.visual.uv) if tri.visual.uv is not None else None
            mat = getattr(tri.visual, "material", None)
            if mat is not None:
                try:
                    tex_img = mat.image  # 优先：内嵌图
                except Exception:
                    tex_img = None
                if tex_img is None:
                    try:
                        tex_img = mat.to_image()  # 兜底导出
                    except Exception:
                        tex_img = None

        if uv is not None and tex_img is not None:
            # 处理UV与贴图
            uv = uv.astype(np.float32).copy()
            if flip_v:
                uv[:, 1] = 1.0 - uv[:, 1]

            if not isinstance(tex_img, np.ndarray):
                tex_img = np.asarray(tex_img.convert("RGBA"))
            if tex_img.shape[-1] == 4:
                tex_img = tex_img[..., :3]
            tex = (tex_img.astype(np.float32) / 255.0)
            # sRGB -> linear（更像Blender）
            tex = np.where(tex <= 0.04045, tex / 12.92, ((tex + 0.055) / 1.055) ** 2.4)

            verts_list.append(V)
            faces_list.append(F)
            verts_uvs_list.append(torch.tensor(uv, dtype=torch.float32, device=device))
            faces_uvs_list.append(F)
            maps_list.append(torch.tensor(tex, dtype=torch.float32, device=device))
        else:
            # --- 兜底：这个子网格没有UV或没有纹理图，但仍要创建一个UV贴图 ---
            # 虚拟 UV（全部 0），1x1 纯白贴图；这样能统一用 TexturesUV，不会把网格丢掉
            verts_list.append(V)
            faces_list.append(F)
            verts_uvs_list.append(torch.zeros((V.shape[0], 2), dtype=torch.float32, device=device))
            faces_uvs_list.append(F)
            maps_list.append(torch.ones((1, 1, 3), dtype=torch.float32, device=device))  # 纯白
    print(f"[UV Loader] collected {len(verts_list)} submeshes "
      f"(maps={len(maps_list)}, uvs={len(verts_uvs_list)}) from GLB")
    if len(verts_list) == 0:
        raise ValueError("UV loader collected 0 submeshes (no verts). "
                     "Check GLB contents or fall back to legacy loader.")
    if len(verts_list) == 0:
        print("[UV Loader] Fallback to legacy loader (no UV/texture found).")
        return load_mesh_any(path, device=device, albedo=0.7)  # 你的旧加载器

    # 把所有带纹理的子网格打包成一个 Meshes
    textures = TexturesUV(
        maps=maps_list,
        faces_uvs=faces_uvs_list,
        verts_uvs=verts_uvs_list,
    )
    mesh = Meshes(verts=verts_list, faces=faces_list, textures=textures)
    return mesh

# ------------------------------ Cameras ------------------------------

def compute_fit_distance(mesh: Meshes, fov_deg: float, margin: float = 1.2) -> float:
    """Estimate camera distance so the object fits in view based on bbox radius and vertical FoV."""
    if len(mesh.verts_list()) == 0:
        raise ValueError("Empty Meshes: no verts. Check loader / GLB contents.")
    with torch.no_grad():
        V = mesh.verts_list()[0]
        vmin = V.min(dim=0).values
        vmax = V.max(dim=0).values
        # center = 0.5 * (vmin + vmax)
        # radius = torch.norm(vmax - center).item()
        extent = (vmax - vmin) * 0.5
        radius = extent.norm().item()          # 原来
        radius_alt = extent.max().item()       # 新方法：最长边的一半
       
    radius = max(radius_alt, 1e-4)
    fov = math.radians(float(fov_deg))
    return max(float((radius / math.tan(0.5 * fov)) * margin), 0.8) # at least 0.8m away to avoid near plane issues

def _zup_eyes_from_spherical(dist, elevs, azims, device):
    """
    Z-up 语义：azim 绕 +Z，elev 相对 +Z（0°=水平，+90°=从正Z俯视）
    返回 eye (N,3)
    """
    import torch
    to_rad = torch.pi / 180.0
    az = torch.as_tensor(azims, dtype=torch.float32, device=device).view(-1) * to_rad
    el = torch.as_tensor(elevs, dtype=torch.float32, device=device).view(-1) * to_rad
    d  = torch.as_tensor(dist,  dtype=torch.float32, device=device).view(-1)
    cx = d * torch.cos(el) * torch.cos(az)
    cy = d * torch.cos(el) * torch.sin(az)
    cz = d * torch.sin(el)
    return torch.stack([cx, cy, cz], dim=-1)  # (N,3)


def ring_equatorial(num: int, elev_deg: float, dist: float, device: torch.device, start_azim_deg: float | None, seed: int | None, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if start_azim_deg is None and seed is not None:
        random.seed(seed)
        start_azim_deg = random.uniform(0.0, 360.0)
    if start_azim_deg is None:
        start_azim_deg = 0.0
    azims = torch.arange(max(num, 1), device=device, dtype=torch.float32) * (360.0 / max(num, 1))
    azims = (azims + float(start_azim_deg)) % 360.0
    elevs = torch.full((num,), float(elev_deg), device=device)
    d = torch.full((num,), float(dist), device=device)
    # R, T = look_at_view_transform(dist=d, elev=elevs, azim=azims, up=((0.0, 1.0, 0.0),))   # ★ 世界Z为up, up points the x-axis
    # 定义相机看向的目标点
    at = (center.view(1,3).expand(num, -1) if center is not None
          else torch.zeros(num, 3, device=device))
    # --- 核心修改 ---
    # 1. 手动计算 Z-up 坐标系下的相机位置 (eye)
    #    注意：_zup_eyes_from_spherical 计算的是围绕原点的相机位置
    eyes_around_origin = _zup_eyes_from_spherical(d, elevs, azims, device)
    # 2. 将相机轨道平移到以物体的中心点 `at` 为中心
    camera_positions = eyes_around_origin + at
    R, T = look_at_view_transform(eye=camera_positions,  # ★ 直接指定 eye
                                  up=((0.0, 0.0, 1.0),),   # ★ Z-up
                                  at=at)                   # ★ 看中心
    return R, T, azims, elevs, d


def ring_top(num: int, top_elev_deg: float, dist: float, device: torch.device, start_azim_deg: float | None, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if start_azim_deg is None:
        start_azim_deg = 0.0
    azims = torch.arange(max(num, 1), device=device, dtype=torch.float32) * (360.0 / max(num, 1))
    azims = (azims + float(start_azim_deg)) % 360.0
    elevs = torch.full((num,), float(top_elev_deg), device=device)
    d = torch.full((num,), float(dist), device=device)
    # R, T = look_at_view_transform(dist=d, elev=elevs, azim=azims, up=((0.0, 1.0, 0.0),))  # ★ 世界Z为up
    at = (center.view(1,3).expand(num, -1) if center is not None
          else torch.zeros(num, 3, device=device))
    # --- 核心修改 ---
    # 1. 手动计算 Z-up 坐标系下的相机位置 (eye)
    #    注意：_zup_eyes_from_spherical 计算的是围绕原点的相机位置
    eyes_around_origin = _zup_eyes_from_spherical(d, elevs, azims, device)
    # 2. 将相机轨道平移到以物体的中心点 `at` 为中心
    camera_positions = eyes_around_origin + at
    R, T = look_at_view_transform(eye=camera_positions,  # ★ 直接指定 eye
                                  up=((0.0, 0.0, 1.0),),   # ★ Z-up
                                  at=at)                   # ★ 看中心
    return R, T, azims, elevs, d


def ring_latitudinal(num: int, azim_deg: float, dist: float, device: torch.device, min_elev_deg: float, max_elev_deg: float, center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Elevation sweep at fixed azimuth (simulate rotating around a horizontal axis)
    if num <= 1:
        elevs = torch.tensor([(min_elev_deg + max_elev_deg) * 0.5], device=device)
    else:
        step = (max_elev_deg - min_elev_deg) / float(num)
        elevs = torch.arange(num, device=device, dtype=torch.float32) * step + (min_elev_deg + 0.5 * step)
    azims = torch.full((num,), float(azim_deg), device=device)
    d = torch.full((num,), float(dist), device=device)
    # R, T = look_at_view_transform(dist=d, elev=elevs, azim=azims, up=((0.0, 1.0, 0.0),))   # ★ 世界Z为up
    at = (center.view(1,3).expand(num, -1) if center is not None
          else torch.zeros(num, 3, device=device))
    # --- 核心修改 ---
    # 1. 手动计算 Z-up 坐标系下的相机位置 (eye)
    #    注意：_zup_eyes_from_spherical 计算的是围绕原点的相机位置
    eyes_around_origin = _zup_eyes_from_spherical(d, elevs, azims, device)
    # 2. 将相机轨道平移到以物体的中心点 `at` 为中心
    camera_positions = eyes_around_origin + at
    R, T = look_at_view_transform(eye=camera_positions,  # ★ 直接指定 eye
                                  up=((0.0, 0.0, 1.0),),   # ★ Z-up
                                  at=at)                   # ★ 看中心
    return R, T, azims, elevs, d


def concat_cameras(Rs: List[torch.Tensor], Ts: List[torch.Tensor], device: torch.device, fov_deg: float) -> FoVPerspectiveCameras:
    R = torch.cat([r for r in Rs if r.numel() > 0], dim=0) if len(Rs) else torch.empty((0, 3, 3), device=device)
    T = torch.cat([t for t in Ts if t.numel() > 0], dim=0) if len(Ts) else torch.empty((0, 3), device=device)
    if R.shape[0] == 0:
        raise ValueError("No cameras generated.")
    # fov can be scalar for all
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov_deg),znear=0.05, zfar=50.0)
    return cameras

# def sample_random_sphere_cameras(
#     num: int,
#     dist_min: float,
#     dist_max: float,
#     device: torch.device,
#     center: torch.Tensor,
#     *,
#     pos_jitter_ratio: float = 0.02,
#     lookat_jitter_ratio: float = 0.01,
#     depth_jitter_ratio: float = 0.02
# ):
#     """
#     在单位球面均匀采样方向 u（高斯→归一化），距离 d~U[min,max]，eye = center + d*u。
#     然后对 eye/at/d 分别加入小噪声（相对尺度），最后用 look_at_view_transform 生成 R,T。
#     """
#     import torch
#     assert dist_max > dist_min > 0
#     # 1) 均匀球面方向（正态→归一化）
#     u = torch.randn(num, 3, device=device)
#     u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # (N,3)
#     # 2) 距离
#     d = torch.empty(num, device=device).uniform_(float(dist_min), float(dist_max))  # (N,)
#     # 3) 位置噪声（与 d 成比例）
#     pos_jit = (pos_jitter_ratio * d).view(-1, 1) * torch.randn(num, 3, device=device)
#     # 4) eye（带位置扰动）
#     eye = center.view(1,3) + u * d.view(-1,1) + pos_jit
#     # 5) look-at 目标点扰动（相对物体尺度）
#     #   用 bbox 尺寸估尺度：这里复用 center 的邻域尺度（取 0.5*(bmax-bmin).norm()）
#     #   → 我们在调用处预先把 half_extent_norm 算好传进来也可以，这里简单给个标量 1.0 占位
#     #   实际在 render_single 里会用真实尺度替换
#     at = center.view(1,3).expand(num, -1)  # 基础 at
#     # 返回 eye/at/d，把 look_at 放到调用处，那里知道物体尺度
#     return eye, at, d
def sample_fixed_sphere_cameras(
    num: int,
    dist: float,                    # 基础半径（相机到中心的距离）
    center: torch.Tensor,           # [3], 物体中心（世界坐标）
    device: torch.device,
    *,
    pos_tangent_jitter: float = 0.02,  # 切向抖动（相对 dist 的比例）
    depth_jitter: float = 0.02         # 径向抖动（相对 dist 的比例）
):
    """
    1) 在单位球面上均匀采方向 u；
    2) 基础相机 eye0 = center + dist * u；
    3) 在 u 的切向平面内加小抖动（幅度≈pos_tangent_jitter * dist）；
    4) 半径做小的相对扰动（幅度≈depth_jitter * dist）；
    5) at 恒为 center（不做 look-at 噪声）。
    """
    # 均匀球面方向（高斯归一化）
    u = torch.randn(num, 3, device=device)
    u = torch.nn.functional.normalize(u, dim=-1, eps=1e-8)  # (N,3)

    # 构造每个 u 的切向正交基 (t1, t2)
    # 选一个与 u 不共线的向量做交叉
    ref = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(u)
    colinear = (u.abs().matmul(ref[0].new_tensor([[0],[0],[1]]).squeeze()) > 0.999).squeeze(-1)
    # ref[colinear] = torch.tensor([0.0, 1.0, 0.0], device=device)  # 避免与 u 近平行
    ref = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(u.shape[0], 1)
    t1 = torch.nn.functional.normalize(torch.cross(u, ref, dim=-1), dim=-1, eps=1e-8)
    t2 = torch.cross(u, t1, dim=-1)  # 已经正交

    # 切向抖动：在 (t1,t2) 平面做 2D 高斯
    tang_amp = (pos_tangent_jitter * dist)
    eps1 = torch.randn(num, 1, device=device)
    eps2 = torch.randn(num, 1, device=device)
    tang = tang_amp * (eps1 * t1 + eps2 * t2)  # (N,3)

    # 径向（深度）抖动
    dr = (depth_jitter * dist) * torch.randn(num, 1, device=device)

    # 眼睛位置
    eye = center.view(1,3) + (dist + dr) * u + tang
    at  = center.view(1,3).expand_as(eye)      # 恒看中心
    return eye, at

def build_lookat_from_eye_at(
    eye: torch.Tensor, at: torch.Tensor, device: torch.device, up=(0.0, 0.0, 1.0)
):
    R, T = look_at_view_transform(eye=eye, at=at, up=(up,), device=device)
    return R, T

# ===== NOCS consistency checker =====
class NOCSConsistencyChecker:
    def __init__(self, voxel=0.02, stride=8, topk=5):
        import numpy as _np
        self.voxel = float(max(voxel, 1e-4))
        self.stride = int(max(stride, 1))
        self.topk = int(max(topk, 1))
        self.grid = {}   # key=(ix,iy,iz) -> {'min':(3,), 'max':(3,), 'count':int}
        self.frame_stats = []
        self._np = _np

    @staticmethod
    def _normalize(P, bmin, span, center, scale_max, mode: str, equal_axis: bool):
        if mode == "bbox":
            return (P - bmin.view(1,1,1,3)) / span.view(1,1,1,3)
        else:  # center_scale
            if equal_axis:
                s = torch.clamp(scale_max, min=1e-8)
                return ((P - center.view(1,1,1,3)) / s) * 0.5 + 0.5
            else:
                return (P - center.view(1,1,1,3)) / torch.clamp(span, min=1e-8).view(1,1,1,3) * 0.5 + 0.5

    def update_batch(self, fragments, mesh_chunk, Xn_est, bmin, bmax, center, scale_max, mode, equal_axis):
        # 片元插值出物体坐标 P_obj（相机无关）
        pix2face = fragments.pix_to_face[..., 0]       # (B,H,W)
        bary     = fragments.bary_coords[..., 0, :]    # (B,H,W,3)
        faces = mesh_chunk.faces_packed()              # (F,3)
        verts = mesh_chunk.verts_packed()              # (V,3)

        fidx = torch.clamp(pix2face, min=0)
        tri  = faces[fidx]                             # (B,H,W,3)
        v0, v1, v2 = verts[tri[...,0]], verts[tri[...,1]], verts[tri[...,2]]
        P_obj = (bary[...,:1]*v0 + bary[...,1:2]*v1 + bary[...,2:]*v2)  # (B,H,W,3)

        valid = (pix2face >= 0)
        if self.stride > 1:
            P_obj   = P_obj[:, ::self.stride, ::self.stride, :]
            Xn_est  = Xn_est[:, ::self.stride, ::self.stride, :]
            valid   = valid[:, ::self.stride, ::self.stride]

        span = (bmax - bmin)
        Xn_ref = self._normalize(P_obj, bmin, span, center, scale_max, mode, equal_axis).clamp(0,1)

        # 逐帧误差（和参考 NOCS 的 L1）
        diff = (Xn_est - Xn_ref).abs()
        B = Xn_est.shape[0]
        for b in range(B):
            vm = valid[b]
            if vm.any():
                d = diff[b][vm]
                # mean / p95 / max（不引入 torch.quantile，兼容性更好）
                mean_L1 = float(d.mean().item())
                # 近似 p95：转到 CPU 用 numpy 百分位（样本量已下采样，不会太慢）
                arr = d.detach().cpu().numpy().reshape(-1, 3)
                p95_L1 = float(self._np.percentile(self._np.mean(arr, axis=1), 95))
                max_L1 = float(d.max().item())
                self.frame_stats.append({'mean_L1':mean_L1, 'p95_L1':p95_L1, 'max_L1':max_L1})
            else:
                self.frame_stats.append({'mean_L1':0.0, 'p95_L1':0.0, 'max_L1':0.0})

        # 跨帧体素聚合：按 Xn_ref 量化，把不同帧的“同一物体点”放到同一桶里
        keys = torch.floor(Xn_ref / self.voxel).to(torch.int32)  # (B,h,w,3)
        kb = keys[valid].view(-1, 3).cpu().numpy()
        vc = Xn_est[valid].view(-1, 3).detach().cpu().numpy()
        for k, val in zip(map(tuple, kb), vc):
            slot = self.grid.get(k)
            if slot is None:
                self.grid[k] = {'min': val.copy(), 'max': val.copy(), 'count': 1}
            else:
                slot['min'] = self._np.minimum(slot['min'], val)
                slot['max'] = self._np.maximum(slot['max'], val)
                slot['count'] += 1

    def finalize(self, topk=None):
        if not self.grid:
            print("[NOCS-Check] No samples accumulated.")
            return {}
        rng_mean = [float((self._np.abs(v['max'] - v['min'])).mean()) for v in self.grid.values()]
        rng_max  = [float((self._np.abs(v['max'] - v['min'])).max())  for v in self.grid.values()]
        mean_range = float(self._np.mean(rng_mean))
        p95_range  = float(self._np.percentile(rng_mean, 95))
        worst_mean = float(self._np.max(rng_mean))
        worst_max  = float(self._np.max(rng_max))
        print(f"[NOCS-Check] Inter-frame L1 range over voxels:"
              f" mean={mean_range:.4g}, p95={p95_range:.4g},"
              f" worst-mean={worst_mean:.4g}, worst-max={worst_max:.4g}"
              f" (voxel={self.voxel}, stride={self.stride})")

        # 打印逐帧 Top-K
        fs = self.frame_stats
        if fs:
            import numpy as _np
            means = _np.array([f['mean_L1'] for f in fs])
            idx = means.argsort()[::-1][:int(topk or self.topk)]
            print("[NOCS-Check] Top frames by mean|X_est - X_ref|:")
            for rk, i in enumerate(idx, 1):
                f = fs[int(i)]
                print(f"  #{rk}: frame {int(i):04d} :: mean={f['mean_L1']:.4g}, p95={f['p95_L1']:.4g}, max={f['max_L1']:.4g}")
        # ...已有打印...
        return {'mean_range':mean_range, 'p95_range':p95_range, 'worst_mean':worst_mean, 'worst_max':worst_max,
            'top_idx': idx.tolist()}


# ------------------------------ Rendering ------------------------------

def make_renderer(cameras: FoVPerspectiveCameras, image_size: int, device: torch.device, *, cull_backfaces: bool, bg_color: Tuple[float, float, float], bin_size: int | None = 0, max_faces_per_bin: int | None = None) -> MeshRenderer:
    # Build raster settings with robust defaults; bin_size=0 uses naive rasterization (no binning), avoids overflow warnings
    rs_kwargs = dict(image_size=int(image_size), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=bool(cull_backfaces))
    if bin_size is not None: rs_kwargs["bin_size"] = int(bin_size)
    if max_faces_per_bin is not None: rs_kwargs["max_faces_per_bin"] = int(max_faces_per_bin)
    # raster_settings = RasterizationSettings(**rs_kwargs)
    # raster_settings = RasterizationSettings(
    #     image_size=image_size,
    #     blur_radius=0.0,
    #     faces_per_pixel=1,      # 1
    #     cull_backfaces=False,   # 看杯子内壁要关掉
    #     bin_size=0,            # 大网格可调大bin提升速度/稳定性 0, automatically chunking
    #     max_faces_per_bin=20000,    #0
    # ) 
    raster_settings = RasterizationSettings(**rs_kwargs)
    lights = PointLights(device=device, location=[(2.0, 2.0, 2.0)], ambient_color=((0.4,0.4,0.4),))
    materials = Materials(
        device=device,
        ambient_color=((0.6, 0.6, 0.6),),
        diffuse_color=((0.9, 0.9, 0.9),),
        specular_color=((0.04, 0.04, 0.04),),
        shininess=32.0,
    )
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights,
                            materials=materials, blend_params=blend),
    )
    return renderer

def _unproject_points_compat(cams: FoVPerspectiveCameras, pts_ndc: torch.Tensor, *, world_coordinates: bool) -> torch.Tensor:
    """Compatibility wrapper for PyTorch3D versions: tries in_ndc then from_ndc."""
    try:
        return cams.unproject_points(pts_ndc, in_ndc=True, world_coordinates=world_coordinates)
    except TypeError:  # pragma: no cover
        return cams.unproject_points(pts_ndc, from_ndc=True, world_coordinates=world_coordinates)

from functools import lru_cache
from pathlib import Path
import numpy as np

def _write_ply_xyzrgb(path, pts, rgb=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    N = int(pts.shape[0])
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is not None and rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        for i in range(N):
            if rgb is None:
                f.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]}\n")
            else:
                r,g,b = int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])
                f.write(f"{pts[i,0]} {pts[i,1]} {pts[i,2]} {r} {g} {b}\n")


@lru_cache(maxsize=8)
def _cached_ndc_grid(H: int, W: int, device_type: str):
    xs_1d = ( (torch.arange(W, device=device_type, dtype=torch.float32) + 0.5) / float(W) ) * 2.0 - 1.0
    ys_1d = 1.0 - ( (torch.arange(H, device=device_type, dtype=torch.float32) + 0.5) / float(H) ) * 2.0
    xs = xs_1d.view(1, 1, W)
    ys = ys_1d.view(1, H, 1)
    return xs, ys

def mesh_aabb_center_scale(mesh):
    Vs = torch.cat(mesh.verts_list(), dim=0)   # ← 聚合所有子网格顶点
    bmin = Vs.min(dim=0).values
    bmax = Vs.max(dim=0).values
    center = 0.5 * (bmin + bmax)
    extent = (bmax - bmin)
    scale_max = extent.max()
    return bmin, bmax, center, scale_max

def render_rgbd_batched(
    mesh: Meshes,
    cameras: FoVPerspectiveCameras,
    image_size: int,
    device: torch.device,
    *,
    batch_chunk: int = 0,
    cull_backfaces: bool = True,
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    bin_size: int | None = 0,
    max_faces_per_bin: int | None = None,
    # --- NEW ---
    return_nocs: bool = True,      # 是否渲染 NOCS
    nocs_mode: str = "bbox",           # 'bbox' or 'center_scale'
    nocs_equal_axis: bool = False,     # center_scale 时是否用最长边等比
     # --- NEW (checker) ---
    check_nocs: bool = False,
    nocs_check_stride: int = 8,
    nocs_check_voxel: float = 0.02,
    nocs_check_topk: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Returns:
      rgb:   [N,H,W,3] in [0,1]
      depth: [N,H,W]   in meters
      nocs:  [N,H,W,3] in [0,1] or None
    """
    N = cameras.R.shape[0]
    if batch_chunk is None or batch_chunk <= 0:
        batch_chunk = N

    meshN = mesh.to(device)

    # 复用的渲染组件（相机相关的每个 chunk 再建）
    rs_kwargs = dict(
        image_size=int(image_size),
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=bool(cull_backfaces),
    )
    if bin_size is not None:
        rs_kwargs["bin_size"] = int(bin_size)
    if max_faces_per_bin is not None:
        rs_kwargs["max_faces_per_bin"] = int(max_faces_per_bin)
    raster_settings = RasterizationSettings(**rs_kwargs)

    lights = PointLights(device=device, location=[(2.0, 2.0, 2.0)], ambient_color=((0.4,0.4,0.4),))
    materials = Materials(
        device=device,
        ambient_color=((0.6, 0.6, 0.6),),
        diffuse_color=((0.9, 0.9, 0.9),),
        specular_color=((0.04, 0.04, 0.04),),
        shininess=32.0,
    )
    blend = BlendParams(background_color=tuple(bg_color))

    rgbs: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    # nocs_list: list[torch.Tensor] | None = [] if return_nocs else None
    nocs_norm_list: list[torch.Tensor] | None = [] if return_nocs else None
    nocs_plus_list: list[torch.Tensor] | None = [] if return_nocs else None
    masks: list[torch.Tensor] = []

    bmax = 0; bmin = 0; center = 0; scale_max = 0;    
    # 预取 AABB/中心/尺度（物体坐标==世界坐标，本脚本就是 canonical object at origin, get before into rendering, then every object not change over different cams
    if return_nocs:
        bmin, bmax, center, scale_max = mesh_aabb_center_scale(mesh.to(device))
        # 避免除零
        eps = torch.tensor(1e-8, device=device, dtype=torch.float32)
        span = torch.clamp(bmax - bmin, min=1e-8)
    
        # extent = bmax - bmin                               
        # tol_center = 1e-4
        # tol_box    = 1e-4

        # # 条件：中心接近 0 且 整体落在 [-0.5, 0.5]^3（允许微小余量）
        # is_centered   = bool((center.abs().max() <= tol_center).item())
        # inside_unit   = bool(((bmin >= (-0.5 - tol_box)).all() and (bmax <= (0.5 + tol_box)).all()).item())
        # use_plus_half = is_centered and inside_unit

        # print(f"[GEOM] center={center.tolist()} extent={extent.tolist()} | "
        #     f"is_centered={is_centered} inside_unit={inside_unit} -> use_plus_half={use_plus_half}")

    checker = None
    if return_nocs and check_nocs:
        checker = NOCSConsistencyChecker(
            voxel=float(nocs_check_voxel),
            stride=int(nocs_check_stride),
            topk=int(nocs_check_topk)
        )
    with torch.inference_mode():
        for i in range(0, N, batch_chunk):
            j = min(i + batch_chunk, N)
            bsz = j - i

            cams_chunk = FoVPerspectiveCameras(
                device=device,
                R=cameras.R[i:j], T=cameras.T[i:j],
                fov=(cameras.fov[i:j] if torch.is_tensor(cameras.fov) else float(cameras.fov)),
                znear=(cameras.znear[i:j] if torch.is_tensor(cameras.znear) else float(cameras.znear)),
                zfar=(cameras.zfar[i:j] if torch.is_tensor(cameras.zfar) else float(cameras.zfar)),
            )


            # === 让 mesh 的 batch 与本 chunk 的相机数一致 ===
            bsz = j - i
            n_mesh = _n_meshes(meshN)

            if n_mesh == 1:
                mesh_chunk = _extend_meshes(meshN, bsz)                 # 单 mesh → 批大小
            elif n_mesh >= j:
                mesh_chunk = meshN[i:j]                                  # 已经是批；切片即可
            elif n_mesh == bsz:
                mesh_chunk = meshN                                       # 批大小刚好匹配
            else:
                raise ValueError(f"Meshes batch ({n_mesh}) not compatible with camera batch ({bsz}).")

            rasterizer = MeshRasterizer(cameras=cams_chunk, raster_settings=raster_settings)
            shader = SoftPhongShader(device=device, cameras=cams_chunk,
                                     lights=lights, materials=materials, blend_params=blend)

            # 栅格化，溢出则退回 naive
            try:
                fragments = rasterizer(mesh_chunk)
            except RuntimeError as e:
                msg = str(e).lower()
                if ("overflow" in msg) or ("bin size was too small" in msg) or ("too many faces" in msg):
                    rasterizer_fallback = MeshRasterizer(
                        cameras=cams_chunk,
                        raster_settings=RasterizationSettings(
                            image_size=int(image_size), blur_radius=0.0, faces_per_pixel=1,
                            cull_backfaces=bool(cull_backfaces), bin_size=0,
                            max_faces_per_bin=int(max_faces_per_bin) if max_faces_per_bin is not None else 0,
                        ),
                    )
                    fragments = rasterizer_fallback(mesh_chunk)
                else:
                    raise
            B = fragments.pix_to_face.shape[0]
            assert len(mesh_chunk) == B, f"Batch mismatch: meshes={len(mesh_chunk)} vs fragments={B}"        
            # 着色用 AMP（只影响 RGB，深度仍保持 float32）
            use_amp = (device.type == 'cuda')
            with torch.cuda.amp.autocast(enabled=use_amp):
                images = shader(fragments, mesh_chunk)  # 用 mesh_chunk (B,H,W,4)
                rgb = images[..., :3].clamp(0, 1)

            zbuf = fragments.zbuf[..., 0].to(torch.float32)  # (B,H,W)
            pix2face = fragments.pix_to_face[..., 0]
            valid = pix2face >= 0

            B, Hc, Wc = zbuf.shape
            xs, ys = _cached_ndc_grid(Hc, Wc, device.type)
            xs = xs.expand(B, Hc, Wc)
            ys = ys.expand(B, Hc, Wc)
            ndc = torch.stack([xs, ys, zbuf], dim=-1).view(B, -1, 3)

            pts_cam = _unproject_points_compat(cams_chunk, ndc, world_coordinates=False)
            depth = pts_cam[..., 2].view(B, Hc, Wc)
            depth = torch.where(valid, depth, torch.zeros_like(depth))
            depth = torch.where(depth > 0, depth, torch.zeros_like(depth))
            mask = valid
                
            # RGB/Depth 也转 CPU，进一步省显存
            rgbs.append(rgb.detach().to('cpu'))       # ← 原来是 .detach()
            depths.append(depth.detach().to('cpu'))   # ← 原来是 .detach()
            masks.append(mask.detach().to('cpu'))
            # === NOCS（世界坐标 ≡ 物体坐标）===
            if return_nocs:
                faces = mesh_chunk.faces_packed()      # (F,3)
                verts = mesh_chunk.verts_packed()      # (V,3)
                fidx  = torch.clamp(pix2face, min=0)   # [-1→0占位]
                tri   = faces[fidx]                    # (B,H,W,3) 顶点索引
                v0, v1, v2 = verts[tri[...,0]], verts[tri[...,1]], verts[tri[...,2]]  # (B,H,W,3)
                bary = fragments.bary_coords[..., 0, :]                                # (B,H,W,3)
                P_ref = (bary[...,:1]*v0 + bary[...,1:2]*v1 + bary[...,2:]*v2)         # (B,H,W,3) 物体坐标
                vmask = valid.unsqueeze(-1)
                pw = torch.where(vmask, P_ref, torch.zeros_like(P_ref))  # ← 用 P_ref 作为 NOCS 的 3D 输入

                # === after computing P_ref and having shader rgb ===
                # === save the pref as ply to visualize it whether it matches the original mesh ===
                # try:
                #     from pathlib import Path as _Path
                #     import numpy as _np

                #     _out_root = getattr(globals().get('a', None), 'out', None)
                #     _out_root = _Path(_out_root) if _out_root is not None else _Path.cwd()
                #     _dump_dir = _out_root / "debug_pref"
                #     _dump_dir.mkdir(parents=True, exist_ok=True)

                #     # 密一点：自适应步长；大图也别太大
                #     _stride = max(1, min(Hc, Wc) // 512)   # 768→stride=1, 1024→2, 2048→4 ...
                #     _max_pts = 300_000

                #     faces = mesh_chunk.faces_packed()
                #     verts = mesh_chunk.verts_packed()
                #     vnorm = mesh_chunk.verts_normals_packed() if hasattr(mesh_chunk, "verts_normals_packed") else None

                #     vm = (pix2face >= 0)     # [B,H,W]
                #     for b in range(B):
                #         gidx = i + b  # 全局帧号
                #         vm_s = vm[b, ::_stride, ::_stride]
                #         if not vm_s.any():
                #             continue

                #         # 采样坐标下的 P_ref、颜色、法线
                #         Pref_s = P_ref[b, ::_stride, ::_stride, :][vm_s]                 # [Ns,3]
                #         rgb_s  = rgb[b,  ::_stride, ::_stride, :][vm_s]                  # [Ns,3] in [0,1]
                #         if vnorm is not None:
                #             tri  = faces[torch.clamp(pix2face[b], min=0)]                # [H,W,3]
                #             tri  = tri[::_stride, ::_stride][vm_s]                       # [Ns,3]
                #             n0, n1, n2 = vnorm[tri[:,0]], vnorm[tri[:,1]], vnorm[tri[:,2]]
                #             bary_s = bary[b, ::_stride, ::_stride, :][vm_s]              # [Ns,3]
                #             nrm_s = (bary_s[:, :1]*n0 + bary_s[:, 1:2]*n1 + bary_s[:, 2:]*n2)
                #             nrm_s = torch.nn.functional.normalize(nrm_s, dim=-1)
                #         else:
                #             nrm_s = None

                #         # 限制点数
                #         if Pref_s.shape[0] > _max_pts:
                #             idx = torch.randperm(Pref_s.shape[0], device=Pref_s.device)[:_max_pts]
                #             Pref_s = Pref_s[idx]; rgb_s = rgb_s[idx]
                #             if nrm_s is not None: nrm_s = nrm_s[idx]

                #         pts_np = Pref_s.detach().cpu().numpy()
                #         rgb_np = (rgb_s.detach().cpu().numpy() * 255.0).clip(0,255).astype(_np.uint8)
                #         if nrm_s is None:
                #             _write_ply_xyzrgb(_dump_dir / f"pref_frame_{gidx:04d}.ply", pts_np, rgb_np)
                #         else:
                #             # 可扩展：把法线也写入（如需，用另一个写函数；这里简单先写 XYZRGB）
                #             _write_ply_xyzrgb(_dump_dir / f"pref_frame_{gidx:04d}.ply", pts_np, rgb_np)

                #         print(f"[PREF] saved {_dump_dir / f'pref_frame_{gidx:04d}.ply'} :: {pts_np.shape[0]} pts; stride={_stride}")

                # except Exception as _e:
                #     print(f"[PREF] save failed: {type(_e).__name__}: {_e}")

                # —— NOCS（未 clamp）：规范化版（norm）+ 直接 +0.5 版（plus）——
                # 1) direct +0.5 始终计算（便于对照）
                Xn_plus_unclamped = pw + 0.5

                if nocs_mode == "bbox":
                    Xn_norm_unclamped = (pw - bmin.view(1,1,1,3)) / span.view(1,1,1,3)
                else:
                    if nocs_equal_axis:
                        s = torch.clamp(scale_max, min=1e-8)
                        Xn_norm_unclamped = (pw - center.view(1,1,1,3)) / s + 0.5
                    else:
                        Xn_norm_unclamped = (pw - center.view(1,1,1,3)) / (span.view(1,1,1,3) + eps) + 0.5
                

                LOG_TOL = 1e-4   # 0.01% 才打印
                # —— 计算阶段（两套都统计）——
                with torch.no_grad():
                    over_plus = ((Xn_plus_unclamped < 0) | (Xn_plus_unclamped > 1)).float().mean().item()
                    over_norm = ((Xn_norm_unclamped < 0) | (Xn_norm_unclamped > 1)).float().mean().item()
                    if (over_plus > LOG_TOL) or (over_norm > LOG_TOL):
                        print(f"[NOCS] out-of-range ratio (pre-clamp): plus={over_plus*100:.4f}%, norm={over_norm*100:.4f}%")

             
                # debug to check the frame-wise NOCS consistency
                # —— 提交给一致性检查器（建议用规范化版去检查）——
                if checker is not None:
                    checker.update_batch(
                        fragments=fragments,
                        mesh_chunk=mesh_chunk,
                        Xn_est=Xn_norm_unclamped,   # 用“规范化版”做一致性检查
                        bmin=bmin, bmax=bmax, center=center, scale_max=scale_max,
                        mode=nocs_mode, equal_axis=nocs_equal_axis
                    )
                # 收集
                if nocs_norm_list is not None:
                    nocs_norm_list.append(Xn_norm_unclamped.detach().to('cpu'))   # ← 转 CPU
                if nocs_plus_list is not None:
                    nocs_plus_list.append(Xn_plus_unclamped.detach().to('cpu'))   # ← 转 CPU
                if device.type == 'cuda' and (i // batch_chunk) % 4 == 3:
                    torch.cuda.empty_cache()  # 释放部分显存，避免峰值过高
                                        
    # --- 循环结束后再 cat ---
    if checker is not None:
        checker.finalize(topk=nocs_check_topk)  
    if not rgbs or not depths:
        raise RuntimeError("No render outputs were produced (empty rgbs/depths). "
                           "Check camera batch, chunk size, and mesh loading.")

    rgb_all   = torch.cat(rgbs, dim=0)              # (N,H,W,3)
    depth_all = torch.cat(depths, dim=0)            # (N,H,W)
    mask_all = torch.cat(masks, dim=0) 

    if return_nocs:
        nocs_norm_all = torch.cat(nocs_norm_list, dim=0) if (nocs_norm_list and len(nocs_norm_list)>0) else None
        nocs_plus_all = torch.cat(nocs_plus_list, dim=0) if (nocs_plus_list and len(nocs_plus_list)>0) else None
        nocs_all = {'norm': nocs_norm_all, 'plus': nocs_plus_all}
    else:
        nocs_all = None


    return rgb_all, depth_all, nocs_all, mask_all


# ------------------------------ IO ------------------------------

def save_mask_series(out_dir: Path, mask: torch.Tensor):
    """
    mask: [N,H,W] bool/0-1 (CPU/Tensor)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    import imageio, numpy as np
    m = (mask.to('cpu').numpy().astype(np.uint8) * 255)  # True→255
    N = m.shape[0]
    for i in range(N):
        imageio.imwrite(out_dir / f"mask_{i:04d}.png", m[i])

def save_rgb_depth_series(out_dir: Path, rgb: torch.Tensor, depth: torch.Tensor, *, 
        save_rgb_png: bool, save_metric_depth: bool, save_depth_png16: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    N, H, W, _ = rgb.shape
    rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth.cpu().numpy().astype(np.float32)

    for i in range(N):
        if save_rgb_png:
            imageio.imwrite(out_dir / f"rgb_{i:04d}.png", rgb_np[i])
        if save_metric_depth:
            np.save(out_dir / f"depth_{i:04d}.npy", depth_np[i])
        if save_depth_png16:
            mm = np.clip(depth_np[i] * 1000.0, 0, 65535).astype(np.uint16)
            imageio.imwrite(out_dir / f"depth_{i:04d}.png", mm)


def make_video_from_rgbs(out_path: Path, rgb: torch.Tensor, fps: int = 24):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    writer = imageio.get_writer(str(out_path), fps=int(fps), codec='libx264', quality=8)
    try:
        frames = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close() #流式写视频，降低峰值内存

def make_depth_video_from_depths(out_path: Path, depth: torch.Tensor, *, fps: int = 24, max_meters: float = 0.0):
    """
    把米制深度 [N,H,W] 映射为 8-bit 灰度视频：
    - max_meters > 0: 0~max_meters 线性映射
    - 否则自动用全序列的 99th 百分位作为上限（鲁棒，不被远点异常值拖垮）
    """
    import imageio, numpy as np
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = depth.detach().cpu().numpy().astype(np.float32)  # [N,H,W]
    d[d <= 0] = np.nan
    if max_meters and max_meters > 0:
        cap = float(max_meters)
    else:
        # 自动上限：序列 99th 百分位
        cap = float(np.nanpercentile(d, 99.0)) if np.isfinite(d).any() else 1.0
    cap = max(cap, 1e-3)
    d_norm = np.clip(d / cap, 0, 1)
    d_norm = np.nan_to_num(d_norm, nan=0.0)
    frames = (d_norm * 255.0).astype(np.uint8)  # 灰度
    writer = imageio.get_writer(str(out_path), fps=int(fps), codec='libx264', quality=8)
    try:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    finally:
        writer.close()

def save_poses_json(out_dir: Path, cameras: FoVPerspectiveCameras, ring_meta: List[Dict]):
    R = cameras.R.detach().cpu().numpy()
    T = cameras.T.detach().cpu().numpy()
    poses = []
    offset = 0
    for meta in ring_meta:
        ring_name = meta["name"]
        num = meta["num"]
        azims = meta.get("azims")
        elevs = meta.get("elevs")
        dist = meta.get("dist")
        for i in range(num):
            idx = offset + i
            Ri = R[idx]
            Ti = T[idx]
            Ci = (-Ri.T @ Ti.reshape(3, 1)).reshape(3)
            poses.append({
                "index": int(idx),
                "ring": ring_name,
                "R_world2cam": Ri.tolist(),
                "T_world2cam": Ti.tolist(),
                "cam_center_world": Ci.tolist(),
                "azim_deg": float(azims[i]) if azims is not None else None,
                "elev_deg": float(elevs[i]) if elevs is not None else None,
                "dist": float(dist) if isinstance(dist, (int, float)) else (float(dist[i]) if dist is not None else None),
            })
        offset += num
    with open(out_dir / "poses.json", "w") as f:
        json.dump(poses, f, indent=2)


def save_intrinsics_json(out_dir: Path, image_size: int, fov_deg: float):
    H = W = int(image_size)
    f = 0.5 * H / math.tan(math.radians(float(fov_deg)) / 2.0)
    intr = {
        "image_size": [H, W],
        "fov_deg_vertical": float(fov_deg),
        "fx": float(f),
        "fy": float(f),
        "cx": float((W - 1) / 2.0),
        "cy": float((H - 1) / 2.0),
        "note": "fx/fy from vertical FoV; PyTorch3D FoV is vertical.",
    }
    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump(intr, f, indent=2)

def save_nocs_series(out_dir: Path, nocs, save_npy=False, save_png8=True, save_png16=False):
    if nocs is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_one(tag: str, arr: torch.Tensor):
        N = arr.shape[0]
        LOG_TOL = 1e-4
        with torch.no_grad():
            over = ((arr < 0) | (arr > 1)).float().mean().item()
            if over > LOG_TOL:
                print(f"[NOCS-SAVE] {tag}: out-of-range before clamp = {over*100:.4f}%")
        # NPY：不 clamp（更保真）
        if save_npy:
            nocs_np = arr.detach().cpu().numpy().astype(np.float32)
        # PNG：clamp 后量化
        if save_png8:
            nocs_png8 = (arr.clamp(0,1).detach().cpu().numpy() * 255.0).astype(np.uint8)
        if save_png16:
            nocs_png16 = (arr.clamp(0,1).detach().cpu().numpy() * 65535.0).astype(np.uint16)

        for i in range(N):
            if save_npy:
                np.save(out_dir / f"nocs_{tag}_{i:04d}.npy", nocs_np[i])
            if save_png8:
                imageio.imwrite(out_dir / f"nocs_{tag}_{i:04d}.png", nocs_png8[i])
            if save_png16:
                imageio.imwrite(out_dir / f"nocs16_{tag}_{i:04d}.png", nocs_png16[i])

    # 支持 dict 或 Tensor
    if isinstance(nocs, dict):
        for tag, arr in nocs.items():
            if isinstance(arr, torch.Tensor) and (arr is not None):
                _save_one(tag, arr)
    elif isinstance(nocs, torch.Tensor):
        _save_one("norm", nocs)  # 单 Tensor 默认按“规范化版”保存
    else:
        print("[NOCS-SAVE] Unsupported nocs type:", type(nocs))



def make_nocs_video(out_path: Path, nocs, fps: int = 24):
    """
    nocs: torch.Tensor of shape [N,H,W,3] in [0,1], or dict[str -> Tensor].
    For dict input, this function will create multiple videos, one per key,
    appending the key to the filename (e.g., orbit_nocs_norm.mp4).
    """
    import imageio
    out_path = Path(out_path)

    # If dict -> write one video per entry and return
    if isinstance(nocs, dict):
        for tag, arr in nocs.items():
            if arr is None:
                continue
            tagged = out_path.with_name(out_path.stem + f"_{tag}" + out_path.suffix)
            make_nocs_video(tagged, arr, fps=fps)
        return

    # Tensor path
    assert isinstance(nocs, torch.Tensor), f"Unsupported nocs type: {type(nocs)}"
    N = int(nocs.shape[0])
    # clamp only for image/video writing; keep float domain otherwise
    frames = (nocs.clamp(0, 1).to('cpu').numpy() * 255.0).astype(np.uint8)  # [N,H,W,3]

    # Write as mp4; stream frame-by-frame to avoid big RAM spikes
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', quality=8, macro_block_size=None)
    try:
        for i in range(N):
            writer.append_data(frames[i])
    finally:
        writer.close()
    print(f"[NOCS-VIDEO] wrote {N} frames → {out_path}")

def _intrinsic_from_fov(image_size: int, fov_deg: float) -> np.ndarray:
    H = W = int(image_size)
    f = 0.5 * H / math.tan(math.radians(float(fov_deg)) / 2.0)
    K = np.array([[f, 0, (W - 1) / 2.0],
                  [0, f, (H - 1) / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)
    return K

def save_h5_per_camera(
    out_dir: Path,
    rgb: torch.Tensor,                 # [N,H,W,3], float in [0,1], CPU
    depth: torch.Tensor,               # [N,H,W],   float (meters), CPU
    mask: torch.Tensor,               # [N,H,W],   float (meters), CPU
    nocs: dict | torch.Tensor | None,  # {'norm': [N,H,W,3], 'plus': [N,H,W,3]} (float [0,1]) or Tensor
    cameras: FoVPerspectiveCameras,
    image_size: int,
    fov_deg: float,
    label: str = "",
    *,
    write_plus: bool = True,           # 写入 +0.5 版本
    write_norm: bool = True            # 写入 规范化版本
):
    out_dir.mkdir(parents=True, exist_ok=True)
    N, H, W, _ = rgb.shape
    assert depth.shape == (N, H, W)

    # 准备内参（相同用于所有帧）
    K = _intrinsic_from_fov(image_size, fov_deg)   # [3,3]
    # 提取外参（世界→相机）
    R = cameras.R.detach().cpu().numpy().astype(np.float32)   # [N,3,3]
    T = cameras.T.detach().cpu().numpy().astype(np.float32)   # [N,3]

    # 统一把输入转为 numpy（RGB/NOCS 量化到 uint8；Mask 从 depth>0 得到）
    rgb_u8  = (rgb.clamp(0,1).numpy() * 255.0).astype(np.uint8)       # [N,H,W,3]
    # NOCS 可能是 dict 或 Tensor；在写 h5 前 clamp & 量化
    nocs_norm_u8 = None
    nocs_plus_u8 = None
    if isinstance(nocs, dict):
        if write_norm and (nocs.get('norm') is not None):
            nocs_norm_u8 = (nocs['norm'].clamp(0,1).numpy() * 255.0).astype(np.uint8)
        if write_plus and (nocs.get('plus') is not None):
            nocs_plus_u8 = (nocs['plus'].clamp(0,1).numpy() * 255.0).astype(np.uint8)
    elif isinstance(nocs, torch.Tensor):
        if write_norm:
            nocs_norm_u8 = (nocs.clamp(0,1).numpy() * 255.0).astype(np.uint8)
    else:
        # 没有 NOCS 也允许，只是不写对应数据集
        pass

    mask_bool = mask  # [N,H,W] bool

    # 每帧一个 .h5
    for i in range(N):
        # 外参（4x4）: world→cam
        Ri = R[i]                        # [3,3]
        Ti = T[i].reshape(3, 1)         # [3,1]
        ext44 = np.eye(4, dtype=np.float32)
        ext44[:3, :3] = Ri
        ext44[:3, 3:] = Ti

        h5_path = out_dir / f"cam_{i:04d}.h5"
        with h5py.File(h5_path, 'w') as f:
            # RGB，保留 N 维（N=1）
            f.create_dataset('RGBs', data=rgb_u8[i:i+1], compression="gzip", compression_opts=4)
            # depth
            # f.create_dataset('Depths', data=depth[i:i+1].numpy().astype(np.float32),
            #          compression="gzip", compression_opts=4)
            # NOCS
            grp = f.create_group('NOCs')
            if nocs_norm_u8 is not None:
                grp.create_dataset('norm', data=nocs_norm_u8[i:i+1], compression="gzip", compression_opts=4)
            if nocs_plus_u8 is not None:
                grp.create_dataset('plus', data=nocs_plus_u8[i:i+1], compression="gzip", compression_opts=4)
            # 标签
            f.attrs['label'] = np.string_(label)
            # Mask（bool）
            grpM = f.create_dataset('Masks', data=mask_bool[i:i+1], compression="gzip", compression_opts=4)
            grpM.attrs['from'] = np.string_('depth>0')
            # 外参/内参
            f.create_dataset('extrinsic_world2cam', data=ext44)  # [4,4] float32
            f.create_dataset('intrinsic', data=K)                # [3,3] float32
            # 可选：把相机中心也写进去（世界坐标）
            C = (-Ri.T @ Ti).reshape(3)   # cam center in world
            f.create_dataset('cam_center_world', data=C.astype(np.float32))
            
        # 可选打印
        # print(f"[H5] wrote {h5_path}")
def save_h5_all(
    out_path: Path,
    rgb: torch.Tensor,        # [N,H,W,3] float[0,1] (CPU)
    depth: torch.Tensor,      # [N,H,W]   float (CPU)
    mask: torch.Tensor,       # [N,H,W]   bool  (CPU)
    nocs: dict | torch.Tensor | None,  # 同你现状：{'norm':..., 'plus':...} 或 Tensor
    cameras: FoVPerspectiveCameras,
    image_size: int,
    fov_deg: float,
    label: str,
    *,
    compress: str = 'gzip',
    gzip_level: int = 6,
    shuffle: bool = False,
    mask_bitpack: bool = False,
    nocs_store: str = 'both',
    nocs_auto_tol: float = 1/255.0
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N, H, W, _ = rgb.shape
    # 量化到 uint8（仅 RGB/NOCS；depth 保留 float32；mask 保留 bool/bit）
    rgb_u8 = (rgb.clamp(0,1).numpy() * 255.0).astype(np.uint8)

    # NOCS 选择策略
    n_norm = None; n_plus = None
    def _to_u8(x): return (x.clamp(0,1).numpy() * 255.0).astype(np.uint8)
    if isinstance(nocs, dict):
        has_norm = isinstance(nocs.get('norm'), torch.Tensor) and (nocs['norm'] is not None)
        has_plus = isinstance(nocs.get('plus'), torch.Tensor) and (nocs['plus'] is not None)
        if nocs_store == 'both' or (nocs_store == 'auto' and not (has_norm and has_plus)):
            n_norm = _to_u8(nocs['norm']) if has_norm else None
            n_plus = _to_u8(nocs['plus']) if has_plus else None
        elif nocs_store == 'norm':
            n_norm = _to_u8(nocs['norm']) if has_norm else None
        elif nocs_store == 'plus':
            n_plus = _to_u8(nocs['plus']) if has_plus else None
        # elif nocs_store == 'auto' and has_norm and has_plus:
        #     # 判等（uint8 域）
        #     diff_max = int(np.abs(_to_u8(nocs['norm']) - _to_u8(nocs['plus'])).max())
        #     if diff_max <= int(round(nocs_auto_tol*255)):
        #         n_norm = _to_u8(nocs['norm'])
        #     else:
        #         n_norm = _to_u8(nocs['norm']); n_plus = _to_u8(nocs['plus'])
    elif isinstance(nocs, torch.Tensor):
        n_norm = _to_u8(nocs)  # 单 Tensor 当作规范化版本
    # depth / mask
    depth_np = depth.numpy().astype(np.float32)
    mask_bool = mask.numpy().astype(bool)

    # 相机矩阵批量
    R = cameras.R.detach().cpu().numpy().astype(np.float32)  # [N,3,3]
    T = cameras.T.detach().cpu().numpy().astype(np.float32)  # [N,3]
    ext = np.repeat(np.eye(4, dtype=np.float32)[None, ...], N, axis=0)  # [N,4,4]
    ext[:, :3, :3] = R
    ext[:, :3, 3]  = T
    # cam center（world）
    C = (-np.transpose(R, (0,2,1)) @ T[..., None]).reshape(N,3).astype(np.float32)
    # 内参（统一）
    K = _intrinsic_from_fov(image_size, fov_deg).astype(np.float32)

    # —— 写 H5（单文件）——
    with h5py.File(str(out_path), 'w') as f:
        
        # 过滤器参数
        comp = None if compress=='none' else compress
        kwargs = {}
        if comp == 'gzip':
            kwargs.update(dict(compression='gzip', compression_opts=int(gzip_level), shuffle=bool(shuffle)))
        elif comp == 'lzf':
            kwargs.update(dict(compression='lzf', shuffle=bool(shuffle)))
        # RGB / NOCS / Mask 
        f.create_dataset('RGBs', data=rgb_u8, **kwargs)            # [N,H,W,3] uint8
        if n_norm is not None:
            f.create_dataset('NOCs/norm', data=n_norm, **kwargs)   # [N,H,W,3] uint8
        if n_plus is not None:
            f.create_dataset('NOCs/plus', data=n_plus, **kwargs)   # [N,H,W,3] uint8
        # label
        f.attrs['label'] = np.string_(label)

        if mask_bitpack:
            # 打包成 bit：shape=[N,ceil(HW/8)]，并记录 H,W
            HW = H*W
            Mbits = []
            for i in range(N):
                Mbits.append(np.packbits(mask_bool[i].reshape(-1).astype(np.uint8), bitorder='big'))
            Mbits = np.stack(Mbits, axis=0)
            ds = f.create_dataset('MasksPacked', data=Mbits, **kwargs)
            ds.attrs['shape'] = np.array([N,H,W], dtype=np.int32)
            ds.attrs['bitorder'] = np.string_('big')
        else:
            f.create_dataset('Masks', data=mask_bool, **kwargs)    # [N,H,W] bool

        # / Depth
        f.create_dataset('Depths', data=depth_np, **kwargs)        # [N,H,W] float32

        # 相机矩阵
        f.create_dataset('extrinsic_world2cam', data=ext)          # [N,4,4] float32
        f.create_dataset('intrinsic', data=K)                      # [3,3]   float32
        f.create_dataset('cam_center_world', data=C)               # [N,3]   float32
# ------------------------------ Driver ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orbit RGBD renderer (PyTorch3D) with multiple camera rings")
    # I/O & base camera
    p.add_argument('--obj', type=str, required=False, help='Path to .glb/.gltf/.obj/.ply (omit when using --manifest)')
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--manifest', type=Path, default=None,
                   help='JSON manifest with entries [{"model": "/path/to/model", "output": "/path/to/out"}]')
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--fov_deg', type=float, default=60.0)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--obj_loader', type=str, default='auto', choices=['auto','trimesh','p3d'],
               help="OBJ 用哪种加载器：auto(默认)/trimesh/p3d(推荐，多材质稳)")
    p.add_argument('--atlas_size', type=int, default=8,
               help="当使用 p3d 加载 OBJ 时，创建纹理图集的分辨率（每面采样大小）")
    p.add_argument('--use_atlas', action='store_true',
               help='为 OBJ 创建 per-face 纹理图集（非常吃内存）。默认关闭，使用 UV 贴图。')
    p.add_argument('--atlas_mem_limit_gb', type=float, default=8.0,
               help='估算 atlas 内存超过该上限则自动改用 UV。默认 2GB。')
    # Equatorial ring (default)
    p.add_argument('--num_cams', type=int, default=80)
    p.add_argument('--elev_deg', type=float, default=20.0)
    p.add_argument('--start_azim_deg', type=float, default=None)
    p.add_argument('--no_random_start', action='store_true')
    # Top ring (overhead) for inside-cup view
    p.add_argument('--top_ring_num', type=int, default=0)
    p.add_argument('--top_ring_elev_deg', type=float, default=75.0)
    p.add_argument('--top_ring_dist_scale', type=float, default=1.0, help='Scale base distance for top ring (e.g., 0.85 closer)')
    # Latitudinal ring (elevation sweep at fixed azimuth)
    p.add_argument('--lat_ring_num', type=int, default=0)
    p.add_argument('--lat_ring_azim_deg', type=float, default=0.0)
    p.add_argument('--lat_ring_min_elev_deg', type=float, default=5.0)
    p.add_argument('--lat_ring_max_elev_deg', type=float, default=175.0)
    p.add_argument('--lat_ring_dist_scale', type=float, default=1.0)
    p.add_argument('--yaw_offset_deg', type=float, default=0.0,
                help='全局偏航（度）。对所有方位角统一加偏移，用于补偿坐标/导出轴向变化（例：180）。')
    p.add_argument('--top_ring_start_azim_deg', type=float, default=None,
                help='顶部环起始方位角；默认与 equatorial 的起始方位（+yaw_offset）保持一致。')
    
    # Loading / appearance
    p.add_argument('--albedo', type=float, default=0.7)
    p.add_argument('--use_uv_textures', action='store_true', help='Use UV baseColor texture if available in GLB')
    p.add_argument('--flip_v', action='store_true', help='Flip V coordinate for UVs (if textures look upside down)')
    p.add_argument('--no_apply_scene_xform', action='store_true', help='Do not bake glTF scene transforms')
    p.add_argument('--axis_correction', type=str, default='none', choices=['none','y_up_to_z_up','z_up_to_y_up'])
    p.add_argument('--no_cull_backfaces', action='store_true', help='Disable backface culling (see inside mug)')
    p.add_argument('--bg_color', type=float, nargs=3, default=[1,1,1], help='Background color RGB in [0,1]')
    # Rasterizer binning (for very dense meshes)
    p.add_argument('--bin_size', type=int, default=64)
    p.add_argument('--max_faces_per_bin', type=int, default=200000)
    # Runtime
    p.add_argument('--batch_chunk', type=int, default=0)
    p.add_argument('--save_rgb_png', action='store_true',
               help='将渲染的 RGB 保存为 PNG（uint8）')
    p.add_argument('--save_mask_png', action='store_true',
               help='将前景 Mask 保存为 PNG（bool → 0/255）')  
    p.add_argument('--save_metric_depth', action='store_true')
    p.add_argument('--save_depth_png16', action='store_true')
    p.add_argument('--make_depth_video', action='store_true', help='把深度序列导出为灰度 mp4（可视化）')
    p.add_argument('--depth_video_max_meters', type=float, default=0.0, help='深度视频上限米数；<=0 则用 99th 百分位自动设定')

    p.add_argument('--make_video', action='store_true')
    p.add_argument('--video_fps', type=int, default=24)
    p.add_argument('--overwrite', action='store_true', help='Remove output directory if it already exists')
    p.add_argument("--input_format", choices=["auto", "glb", "obj"], default="auto",
                help="只处理指定类型的输入：'obj' 仅索引 *_obj 目录；'glb' 仅索引 .glb；'auto' 沿用默认优先级")
    # --- NOCS 输出相关 ---
    p.add_argument('--save_nocs', action='store_true',
                help='保存 NOCS 的 float32 .npy（H,W,3，范围[0,1]）')
    p.add_argument('--save_nocs_png8', action='store_true',
                help='保存 NOCS 的 8-bit PNG 可视化（RGB=xyz∈[0,255]）')
    p.add_argument('--make_nocs_video', action='store_true',
                help='把 NOCS 序列导出为 mp4（RGB=xyz∈[0,255]）')
    p.add_argument('--nocs_norm', choices=['bbox','center_scale'], default='bbox',
                help='NOCS 归一化方式：bbox 线性到 [0,1]^3；或 center_scale 等比缩放到 [-0.5,0.5]^3 再平移到 [0,1]^3')
    p.add_argument('--nocs_equal_axis', action='store_true',
                help='与 center_scale 联用：用最长边作为统一尺度（等比）；否则对每轴分别缩放（与 bbox 等价）')
    p.add_argument('--check_nocs', action='store_true',
              help='在渲染过程中检查 NOCS 跨帧颜色一致性并打印统计')
    p.add_argument('--nocs_check_stride', type=int, default=8,
                help='抽样步长（像素网格步进）；越大越快、越粗略')
    p.add_argument('--nocs_check_voxel', type=float, default=0.02,
                help='参考 NOCS 的体素量化步长，用于跨帧对齐同一物体点（0~1 归一化域）')
    p.add_argument('--nocs_check_topk', type=int, default=8,
                help='打印逐帧误差最大的前 K 帧')
    p.add_argument('--label', type=str, default='',
                   help='类别标签写入 h5；为空则尝试从路径名推断')
    p.add_argument('--save_h5', action='store_true',
                   help='为每个相机视角写一个 .h5，包含 RGB/NOCS/Mask/外参/内参/label')
    # --- 相机模式：rings(默认) / random ---
    p.add_argument('--cam_mode', type=str, default='rings', choices=['rings','random'],
                help='相机生成模式：rings=等间隔轨道；random=球面随机采样')
    # random 采样参数
    p.add_argument('--rand_cams', type=int, default=80, help='random 模式下相机数量')
    p.add_argument('--sphere_dist', type=float, default=-1.0,
               help='球面半径（相机到物体中心的基础距离）；<=0 则自动根据FoV/AABB估计')
    p.add_argument('--pos_tangent_jitter', type=float, default=0.02,
                help='切向位置扰动系数（相对距离的比例），例如 0.02=2%·dist')
    p.add_argument('--depth_jitter', type=float, default=0.02,
                help='径向（深度）扰动系数（相对距离的比例），例如 0.02=2%·dist')      
    return p.parse_args()

def prepare_out_dir(out_dir: str, overwrite: bool) -> Path:
    p = Path(out_dir).resolve()
    if p.exists():
        if overwrite:
            # 安全保护：避免误删根目录等
            if str(p) in ("/", "/home", "/root"):  # 可按需扩充
                raise ValueError(f"Refusing to delete dangerous path: {p}")
            shutil.rmtree(p)
        else:
            raise FileExistsError(f"{p} already exists. Use --overwrite to replace.")
    p.mkdir(parents=True, exist_ok=True)
    return p

# === NEW: render one target (reuses your existing logic) ===
def render_single(a, device, src_path: Path, out_dir: Path):
    # choose loader by file suffix, NOT by input_format
    ext = src_path.suffix.lower()

    # --- load mesh (keep your original branches) ---
    if ext == ".obj" and a.use_uv_textures and a.obj_loader in ("auto", "p3d"):
        try:
            mesh = load_obj_with_textures_p3d(
                src_path,
                device=device,
                use_atlas=a.use_atlas,
                atlas_size=a.atlas_size,
                atlas_mem_limit_gb=a.atlas_mem_limit_gb,
                axis_correction=a.axis_correction,
            )
            print(f"[info] OBJ via PyTorch3D loader: {src_path}")
        except Exception as e:
            print(f"[warn] p3d OBJ loader failed: {e} ; falling back to trimesh loader.")
            mesh = load_mesh_any(
                src_path, device=device,
                albedo=a.albedo, use_uv_textures=a.use_uv_textures,
                flip_v=a.flip_v, apply_scene_xform=(not a.no_apply_scene_xform),
                axis_correction=a.axis_correction,
            )
    else:
        mesh = load_mesh_any(
            src_path, device=device,
            albedo=a.albedo, use_uv_textures=a.use_uv_textures,
            flip_v=a.flip_v, apply_scene_xform=(not a.no_apply_scene_xform),
            axis_correction=a.axis_correction,
        )

    # --- camera rings (unchanged) ---
    base_dist = compute_fit_distance(mesh, fov_deg=a.fov_deg, margin=1.8)
    # 统一的起始方位：把 yaw_offset 叠加到 equatorial / top / lat
    # 注意：a.start_azim_deg=0 要被正确识别为“给定 0°”，不能当成 None
    start0 = a.start_azim_deg if (a.start_azim_deg is not None) else (0.0 if a.no_random_start else None)
    if start0 is not None:
        start0 = float(start0) + float(a.yaw_offset_deg)
    # 顶部环起始方位：显式给则用显式；否则跟随 equatorial（若 equatorial 随机，则用 yaw_offset）
    top_start = a.top_ring_start_azim_deg if (a.top_ring_start_azim_deg is not None) else (start0 if start0 is not None else float(a.yaw_offset_deg))
    # 纬向环固定方位：在用户给定基础上加偏航
    lat_azim = float(a.lat_ring_azim_deg) + float(a.yaw_offset_deg)

    Vs = torch.cat(mesh.verts_list(), dim=0)
    bmin = Vs.min(0).values; bmax = Vs.max(0).values
    center = 0.5*(bmin + bmax)
    bbox_extent = (bmax - bmin)
    obj_scale = float(bbox_extent.norm().item())  # 一个代表性全局尺度

    if a.cam_mode == 'random':
        # 基础半径：优先用户指定，否则自动估计
        base_dist = float(a.sphere_dist)
        if base_dist <= 0:
            base_dist = compute_fit_distance(mesh, fov_deg=a.fov_deg, margin=1.6)  # 比 rings 稍紧凑一点

        # 采样固定球面 + 小扰动（无 look-at 抖动）
        eye, at = sample_fixed_sphere_cameras(
            num=a.rand_cams,
            dist=base_dist,
            center=center,
            device=device,
            pos_tangent_jitter=float(a.pos_tangent_jitter),
            depth_jitter=float(a.depth_jitter),
        )

        # 构建 R,T（Z-up；始终看 center）
        R, T = build_lookat_from_eye_at(eye, at, device=device, up=(0.0, 0.0, 1.0))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(a.fov_deg), znear=0.05, zfar=50.0)

        # 记录 meta（便于复现/审计）
        meta = [{
            "name": "random_fixed_sphere", "num": int(a.rand_cams), "azims": None, "elevs": None, "dist": float(base_dist),
            }]
    else:
        Rs, Ts, meta = [], [], []
        R0, T0, az0, el0, d0 = ring_equatorial(
            num=a.num_cams, elev_deg=a.elev_deg, dist=base_dist, device=device,
            start_azim_deg=start0,
            seed=(None if a.no_random_start else a.seed), center=center
        )
        Rs.append(R0); Ts.append(T0)
        meta.append({"name":"equatorial","num":a.num_cams,
                    "azims":az0.tolist(),"elevs":el0.tolist(),"dist":float(base_dist)})

        if a.top_ring_num > 0:
            top_dist = float(base_dist) * float(a.top_ring_dist_scale)
            R1, T1, az1, el1, d1 = ring_top(
                num=a.top_ring_num, top_elev_deg=a.top_ring_elev_deg,
                dist=top_dist, device=device, start_azim_deg=top_start, center=center,
            )  
            Rs.append(R1); Ts.append(T1)
            meta.append({"name":"top","num":a.top_ring_num,
                        "azims":az1.tolist(),"elevs":el1.tolist(),"dist":float(top_dist)})

        if a.lat_ring_num > 0:
            lat_dist = float(base_dist) * float(a.lat_ring_dist_scale)
            R2, T2, az2, el2, d2 = ring_latitudinal(
                num=a.lat_ring_num, azim_deg=lat_azim, dist=lat_dist, device=device,
                min_elev_deg=a.lat_ring_min_elev_deg, max_elev_deg=a.lat_ring_max_elev_deg,
                center=center,
            )
            Rs.append(R2); Ts.append(T2)
            meta.append({"name":"latitudinal","num":a.lat_ring_num,
                        "azims":az2.tolist(),"elevs":el2.tolist(),"dist":float(lat_dist)})
        # # pytorch 3d convention is y up, z forward (set up camera using look_at_view_transform), so to preserve the obj orientation, we need to swap y and z axis
        # # === 将相机外参从 Y-up 基底重写为 Z-up 基底（只改 R，T 保持不变）===
        # # 依赖你脚本里已有的 axis_correction_matrix
        # Q = axis_correction_matrix("y_up_to_z_up", device=Rs[0].device)  # [3,3]，绕 +X 轴 +90°
        # Rs = [R @ Q for R in Rs]  # 右乘基变换：R_zup = R_yup @ Q
        cameras = concat_cameras(Rs, Ts, device=device, fov_deg=a.fov_deg)

    # --- render & save (unchanged) ---
    rgb, depth, nocs, mask = render_rgbd_batched(
        mesh=mesh, cameras=cameras, image_size=a.image_size, device=device,
        batch_chunk=a.batch_chunk, cull_backfaces=(not a.no_cull_backfaces),
        bg_color=tuple(a.bg_color),
        bin_size=(None if a.bin_size is None else int(a.bin_size)),
        max_faces_per_bin=(None if a.max_faces_per_bin is None else int(a.max_faces_per_bin)),
        # NOCS
        return_nocs = (a.save_nocs or a.save_nocs_png8 or a.make_nocs_video or a.save_h5),
        nocs_mode = a.nocs_norm,
        nocs_equal_axis = a.nocs_equal_axis,
        # NOCS checker
        check_nocs=a.check_nocs,
        nocs_check_stride=a.nocs_check_stride,
        nocs_check_voxel=a.nocs_check_voxel,
        nocs_check_topk=a.nocs_check_topk,
    )
    # --- NEW: 保存 NOCS 帧序列 ---
    if (a.save_nocs or a.save_nocs_png8) and (nocs is not None):
        save_nocs_series(out_dir, nocs, save_npy=a.save_nocs, save_png8=a.save_nocs_png8)

    save_rgb_depth_series(out_dir, rgb, depth,
                          save_rgb_png=a.save_rgb_png,
                          save_metric_depth=a.save_metric_depth,
                          save_depth_png16=a.save_depth_png16)
    save_poses_json(out_dir, cameras, meta)
    save_intrinsics_json(out_dir, a.image_size, a.fov_deg)
    if a.make_video:
        make_video_from_rgbs(out_dir / 'orbit_rgb.mp4', rgb, fps=a.video_fps)
    if a.make_depth_video:
        make_depth_video_from_depths(out_dir / 'orbit_depth.mp4', depth, fps=a.video_fps, max_meters=a.depth_video_max_meters)
    # --- NEW: NOCS 视频 ---
    if a.make_nocs_video and (nocs is not None):
        make_nocs_video(out_dir / 'orbit_nocs.mp4', nocs, fps=a.video_fps)
    if a.save_h5:
        # 注意：我们前面把 rgb/depth 已经转到 CPU；nocs 也是 CPU
        label = (a.label.strip() or out_dir.name.split('_')[0])
        out_h5 = out_dir / "all_cameras.h5"
        save_h5_all(
            out_path=out_h5,
            rgb=rgb, depth=depth, mask=mask, nocs=nocs,
            cameras=cameras, image_size=a.image_size, fov_deg=a.fov_deg, label=label,
            # compress=a.h5_compress, gzip_level=a.h5_gzip_level, shuffle=a.h5_shuffle,
            # mask_bitpack=a.mask_bitpack,  nocs_auto_tol=a.nocs_auto_tol,
            # nocs_store=a.nocs_store,
        )
    if a.save_mask_png:
        # 如果 render_rgbd_batched 已经返回了 mask 张量（见下一节），直接用：
        save_mask_series(out_dir / "mask_png", mask)
import time         
def main():
    a = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("[info] GPU rendering on:", torch.cuda.get_device_name(0))
    set_seed(a.seed)

    debug_print_p3d_capability_once()

    # --- manifest-driven batch mode -------------------------------------------------
    tasks: List[Tuple[Path, Path]] = []
    if a.manifest is not None:
        manifest_path = Path(a.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to read manifest JSON: {e}") from e
        if not isinstance(manifest_data, list):
            raise ValueError("Manifest must be a list of {\"model\":..., \"output\":...}")
        for idx, entry in enumerate(manifest_data):
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest entry #{idx} is not an object")
            if "model" not in entry or "output" not in entry:
                raise ValueError(f"Manifest entry #{idx} missing 'model' or 'output'")
            tasks.append((Path(entry["model"]), Path(entry["output"])))
        if not tasks:
            print("[warn] Manifest is empty; nothing to render.")
            return

        for mesh_path, out_dir_path in tasks:
            if not mesh_path.exists():
                print(f"[ERROR] Manifest model does not exist: {mesh_path}")
                continue
            out_dir = prepare_out_dir(str(out_dir_path), a.overwrite)
            try:
                start = time.time()
                print(f"\n===== Rendering: {mesh_path} → {out_dir} =====")
                render_single(a, device, mesh_path, out_dir)
                end = time.time()
                duration = end - start
                print(f"time: {duration:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Failed: {mesh_path} :: {e}")
        return

    if not a.obj:
        raise ValueError("--obj is required when --manifest is not provided")

    src = Path(a.obj)

    # --- directory mode: enumerate & batch render by input_format ---
    if src.is_dir():
        out_root = Path(a.out).resolve()
        out_root.mkdir(parents=True, exist_ok=True)

        targets = enumerate_targets(src, a.input_format)
        if not targets:
            raise RuntimeError(f"No targets found under {src} with input_format={a.input_format}")
        print(f"[info] Found {len(targets)} targets under {src} (input_format={a.input_format})")

        for mesh_path, name in targets:
            out_dir = prepare_out_dir(str(out_root / name), a.overwrite)
            try:
                
                start = time.time()
                print(f"\n===== Rendering: {mesh_path} → {out_dir} =====")
                render_single(a, device, mesh_path, out_dir)
                end = time.time()
                duration = end - start
                print(f"time: {duration:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Failed: {mesh_path} :: {e}")
        return

    # --- single-file mode: keep original behaviour ---
    if not src.exists():
        raise FileNotFoundError(f"--obj not found: {src}")
    out_dir = prepare_out_dir(a.out, a.overwrite)
    print(f"\n===== Rendering: {src} → {out_dir} =====")
    start = time.time()
    render_single(a, device, src, out_dir)
    end = time.time()
    duration = end - start
    print(f"time: {duration:.2f} seconds")
if __name__ == "__main__":
    main()
