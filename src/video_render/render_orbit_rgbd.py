#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_orbit_rgbd.py — 支持 .glb/.gltf/.obj/.ply 的并行 RGBD 轨道渲染（PyTorch3D）

用法示例：
python render_orbit_rgbd.py \
  --obj ../Data/aligned_mugs/mug_1532_7659420.glb \
  --num_cams 80 --elev_deg 20 --image_size 512 --fov_deg 60 \
  --seed 123 --save_metric_depth --make_video \
  --out ../Data/out_orbit

依赖：
  pip install torch torchvision pytorch3d trimesh imageio imageio-ffmpeg

注意：
  * 该脚本对 .glb/.gltf 使用 trimesh 加载，并自动转换为 PyTorch3D Meshes。
  * 优先使用顶点颜色渲染；若无顶点颜色，则使用统一的中性材质（可通过 --albedo 调整）。
  * 深度图保存为 .npy（米为单位）+ 可选 16-bit PNG 预览（--save_depth_png16）。
"""

from __future__ import annotations
import argparse
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import imageio
import numpy as np
import torch

# PyTorch3D
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    HardPhongShader,
    PointLights,
)

from pytorch3d.renderer.mesh.shader import BlendParams
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer.cameras import look_at_view_transform

# trimesh 用于加载 glb/gltf/ply/obj，并抽取顶点/面/颜色
import trimesh


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_tensor(x, device, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=device)


def _concat_mesh_list(mesh_list: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(mesh_list) == 1:
        return mesh_list[0]
    return trimesh.util.concatenate(mesh_list)


def load_mesh_any(path: str | Path, device: torch.device, albedo: float = 0.7) -> Meshes:
    """加载任意网格（.glb/.gltf/.obj/.ply），输出 PyTorch3D Meshes。

    * 优先使用顶点颜色（TexturesVertex）。
    * 若存在多子网格，合并为单一网格。
    * 目前不做 UV 纹理重建（glTF PBR -> 简化为顶点色/固有色）。
    """
    path = str(path)
    # force='scene' 让 trimesh 统一返回 Scene 或 Trimesh
    scene_or_mesh = trimesh.load(path, force='scene')

    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        if len(geoms) == 0:
            raise ValueError(f"No geometry found in {path}")
        tri = _concat_mesh_list([g.as_trimesh() if hasattr(g, 'as_trimesh') else g for g in geoms])
    else:
        tri = scene_or_mesh

    if not isinstance(tri, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a Trimesh: {type(tri)} from {path}")

    # 顶点、面
    verts = _to_tensor(tri.vertices, device)
    faces = _to_tensor(tri.faces, device, dtype=torch.int64)

    # 顶点颜色（若存在），范围 [0,1]
    if hasattr(tri.visual, 'vertex_colors') and tri.visual.vertex_colors is not None:
        vc = np.asarray(tri.visual.vertex_colors)
        if vc.shape[1] == 4:
            vc = vc[:, :3]
        if vc.max() > 1.0:
            vc = vc / 255.0
        verts_rgb = _to_tensor(vc, device)
    else:
        # 无顶点色则给一个统一的反照率
        verts_rgb = torch.full((verts.shape[0], 3), float(albedo), device=device)

    textures = TexturesVertex(verts_features=[verts_rgb])
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh


def compute_fit_distance(mesh: Meshes, fov_deg: float, margin: float = 1.2) -> float:
    """根据网格包围球和 FOV 估计相机距离，确保完整包裹对象。"""
    with torch.no_grad():
        V = mesh.verts_list()[0]
        vmin = V.min(dim=0).values
        vmax = V.max(dim=0).values
        center = 0.5 * (vmin + vmax)
        radius = torch.norm(vmax - center).item()
    if radius <= 0:
        radius = 1.0
    fov = math.radians(float(fov_deg))
    dist = (radius / math.tan(0.5 * fov)) * margin
    return float(dist)


def build_orbit_cameras(
    num_cams: int,
    elev_deg: float,
    fov_deg: float,
    dist: float,
    device: torch.device,
    start_azim_deg: float = 0.0,
) -> FoVPerspectiveCameras:
    # Older PyTorch doesn't support `endpoint` in linspace; use arange to avoid the 360° duplicate
    azims = torch.arange(num_cams, device=device, dtype=torch.float32) * (360.0 / float(num_cams))
    azims = (azims + start_azim_deg) % 360.0
    d = torch.full((num_cams,), float(dist), device=device)
    e = torch.full((num_cams,), float(elev_deg), device=device)
    R, T = look_at_view_transform(dist=d, elev=e, azim=azims)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=float(fov_deg))
    return cameras


def make_renderer(
    cameras: FoVPerspectiveCameras,
    image_size: int,
    device: torch.device,
) -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=int(image_size),
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True,
    )
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])
    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader,
    )
    return renderer


def render_rgbd_batched(
    mesh: Meshes,
    cameras: FoVPerspectiveCameras,
    image_size: int,
    device: torch.device,
    batch_chunk: int = 0,
    save_metric_depth: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """并行渲染 RGB 与 metric 深度。

    返回：
      rgb:  (N,H,W,3) float32 in [0,1]
      depth:(N,H,W)   float32 (米)
    """
    N = cameras.R.shape[0]
    meshN = mesh.extend(N)

    if batch_chunk is None or batch_chunk <= 0:
        batch_chunk = N

    rgbs = []
    depths = []

    for i in range(0, N, batch_chunk):
        j = min(i + batch_chunk, N)
        cams_chunk = FoVPerspectiveCameras(
            device=device,
            R=cameras.R[i:j],
            T=cameras.T[i:j],
            fov=cameras.fov[i:j] if torch.is_tensor(cameras.fov) else float(cameras.fov),
        )
        renderer = make_renderer(cams_chunk, image_size=image_size, device=device)

        # --- RGB ---
        images = renderer(meshN[i:j])  # (B,H,W,4)
        rgb = images[..., :3]
        rgbs.append(rgb.detach())

        # --- Depth（metric）---
        # 从 rasterizer 拿到 fragments.zbuf (NDC 深度)，然后用 unproject_points 转回相机坐标系 Z
        # --- Depth（metric）---
        # zbuf 为命中像素的 NDC 深度；用 pix_to_face 做有效像素掩码
        fragments = renderer.rasterizer(meshN[i:j])
        zbuf = fragments.zbuf[..., 0]            # (B,H,W)
        pix2face = fragments.pix_to_face[..., 0] # (B,H,W)
        valid = pix2face >= 0

        # 构造 (B,HW,3) 的 NDC 点，再反投影回相机坐标系
        B, H, W = zbuf.shape
        xs = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, W).expand(B, H, W)
        ys = torch.linspace(-1.0, 1.0, H, device=device).view(1, H, 1).expand(B, H, W)
        ndc = torch.stack([xs, ys, zbuf], dim=-1).view(B, -1, 3)  # (B,HW,3)

        pts_cam = cams_chunk.unproject_points(ndc, in_ndc=True, world_coordinates=False)  # (B,HW,3)
        depth = pts_cam[..., 2].view(B, H, W)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        depth = torch.where(depth > 0, depth, torch.zeros_like(depth))

        depths.append(depth.detach())

    rgb_all = torch.cat(rgbs, dim=0)
    depth_all = torch.cat(depths, dim=0)
    return rgb_all, depth_all


def save_rgb_depth_series(
    out_dir: Path,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    save_metric_depth: bool,
    save_depth_png16: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    N, H, W, _ = rgb.shape

    rgb_np = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    depth_np = depth.cpu().numpy().astype(np.float32)

    for i in range(N):
        imageio.imwrite(out_dir / f"rgb_{i:04d}.png", rgb_np[i])
        if save_metric_depth:
            np.save(out_dir / f"depth_{i:04d}.npy", depth_np[i])
        if save_depth_png16:
            # 可视化用途：把米单位缩放到 mm，截断到 16-bit 范围
            mm = np.clip(depth_np[i] * 1000.0, 0, 65535).astype(np.uint16)
            imageio.imwrite(out_dir / f"depth_{i:04d}.png", mm)


def make_video_from_rgbs(out_path: Path, rgb: torch.Tensor, fps: int = 24):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    imageio.mimsave(out_path, frames, fps=int(fps))


def render_orbit(
    obj_path: str,
    num_cams: int,
    elev_deg: float,
    image_size: int,
    fov_deg: float,
    seed: int | None,
    out_dir: str,
    albedo: float = 0.7,
    start_azim_random: bool = True,
    batch_chunk: int = 0,
    save_metric_depth: bool = True,
    save_depth_png16: bool = False,
    make_video: bool = False,
    video_fps: int = 24,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    mesh = load_mesh_any(obj_path, device=device, albedo=albedo)

    # 估计一个合适的轨道距离，确保完整包裹对象
    dist = compute_fit_distance(mesh, fov_deg=fov_deg, margin=1.25)

    start_azim_deg = float(random.uniform(0, 360)) if start_azim_random else 0.0
    cameras = build_orbit_cameras(
        num_cams=num_cams,
        elev_deg=elev_deg,
        fov_deg=fov_deg,
        dist=dist,
        device=device,
        start_azim_deg=start_azim_deg,
    )

    rgb, depth = render_rgbd_batched(
        mesh=mesh,
        cameras=cameras,
        image_size=image_size,
        device=device,
        batch_chunk=batch_chunk,
        save_metric_depth=save_metric_depth,
    )

    out_dir = Path(out_dir)
    save_rgb_depth_series(
        out_dir=out_dir,
        rgb=rgb,
        depth=depth,
        save_metric_depth=save_metric_depth,
        save_depth_png16=save_depth_png16,
    )

    if make_video:
        make_video_from_rgbs(out_dir / 'orbit_rgb.mp4', rgb, fps=video_fps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Orbit RGBD renderer (PyTorch3D, GLB supported).')
    p.add_argument('--obj', type=str, required=True, help='Path to .glb/.gltf/.obj/.ply')
    p.add_argument('--num_cams', type=int, default=80)
    p.add_argument('--elev_deg', type=float, default=30.0)
    p.add_argument('--fov_deg', type=float, default=70.0)
    p.add_argument('--image_size', type=int, default=512)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--out', type=str, required=True)

    p.add_argument('--albedo', type=float, default=0.7, help='No-vertex-color fallback albedo [0,1].')
    p.add_argument('--no_random_start', action='store_true', help='Do not randomize starting azimuth.')
    p.add_argument('--batch_chunk', type=int, default=0, help='Split batch to avoid OOM; 0=all-at-once')

    p.add_argument('--save_metric_depth', action='store_true', help='Save metric depth (.npy meters).')
    p.add_argument('--save_depth_png16', action='store_true', help='Also save 16-bit PNG preview (mm).')
    p.add_argument('--make_video', action='store_true', help='Export mp4 from the N orbit frames.')
    p.add_argument('--video_fps', type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    render_orbit(
        obj_path=args.obj,
        num_cams=args.num_cams,
        elev_deg=args.elev_deg,
        image_size=args.image_size,
        fov_deg=args.fov_deg,
        seed=args.seed,
        out_dir=args.out,
        albedo=args.albedo,
        start_azim_random=(not args.no_random_start),
        batch_chunk=args.batch_chunk,
        save_metric_depth=args.save_metric_depth,
        save_depth_png16=args.save_depth_png16,
        make_video=args.make_video,
        video_fps=args.video_fps,
    )


if __name__ == '__main__':
    main()
