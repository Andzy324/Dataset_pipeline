#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from PIL import Image

import torch
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights, RasterizationSettings,
    MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesUV, TexturesVertex, Materials
)
import trimesh


def _to_tensor(x, device, dtype=torch.float32):
    return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)


def load_glb_as_meshes(
    glb_path: str,
    device: torch.device,
    flip_v: bool = True,
    prefer_vertex_color_if_no_texture: bool = True,
):
    """
    从 .glb 读取并转成一个 PyTorch3D Meshes（合并 scene 内所有子几何）。
    支持：
      - UV+贴图 -> TexturesUV
      - 顶点颜色 COLOR_0 -> TexturesVertex
      - 仅 baseColorFactor -> 把材质色烘为顶点色 -> TexturesVertex
    """
    scene = trimesh.load(glb_path, force='scene')
    if not isinstance(scene, trimesh.Scene):
        # 单一 Trimesh 也包一层
        scene = trimesh.Scene(scene)

    verts_all = []
    faces_all = []
    textures_list = []

    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue

        # 顶点 & 面
        V = np.asarray(geom.vertices, dtype=np.float32)
        F = np.asarray(geom.faces, dtype=np.int64)

        if V.size == 0 or F.size == 0:
            continue

        # 把局部变换应用到几何（scene graph）
        tf = scene.graph.get(name)[0] if name in scene.graph.nodes_geometry else np.eye(4)
        V_h = np.concatenate([V, np.ones((V.shape[0], 1), dtype=np.float32)], axis=1)
        V_w = (V_h @ tf.T)[:, :3]

        # 默认用纯白顶点色（如果后面没贴图/没顶点色/没材质色，会看到白色+阴影）
        colors_v = np.ones((V_w.shape[0], 3), dtype=np.float32)

        used_texture = False
        used_vertex_color = False

        # 1) 如果有 UV 和贴图
        if isinstance(geom.visual, trimesh.visual.texture.TextureVisuals):
            # UV
            uv = np.asarray(geom.visual.uv, dtype=np.float32) if geom.visual.uv is not None else None
            # 贴图图像
            img = None
            mat = getattr(geom.visual, "material", None)
            if mat is not None:
                # baseColorTexture
                tex = getattr(mat, "baseColorTexture", None)
                if tex is not None and getattr(tex, "image", None) is not None:
                    img = tex.image  # PIL Image 或 ndarray
                # 有的导出器把贴图放在 material.image
                if img is None and getattr(mat, "image", None) is not None:
                    img = mat.image

            if uv is not None and img is not None:
                # 归一化 UV；有的导出器 UV 可能已在 [0,1]，有的不是
                uv_use = uv.copy()
                if flip_v:
                    uv_use[:, 1] = 1.0 - uv_use[:, 1]

                # faces_uvs
                if geom.visual.faces is not None and len(geom.visual.faces) == len(F):
                    F_uv = np.asarray(geom.visual.faces, dtype=np.int64)
                else:
                    # 有些 glb 没有单独的 uv faces，退化为按顶点索引
                    F_uv = F.copy()

                # 贴图转 tensor
                if isinstance(img, Image.Image):
                    tex_img = np.asarray(img.convert("RGB"), dtype=np.uint8)
                else:
                    arr = np.asarray(img)
                    if arr.ndim == 2:
                        arr = np.stack([arr, arr, arr], axis=-1)
                    tex_img = arr[..., :3].astype(np.uint8)

                # PyTorch3D 要 HxWx3 float[0,1]
                tex_map = torch.from_numpy(tex_img.astype(np.float32) / 255.0)[None].to(device)
                verts_uvs = torch.from_numpy(uv_use.astype(np.float32)).to(device)
                faces_uvs = torch.from_numpy(F_uv.astype(np.int64)).to(device)

                textures = TexturesUV(
                    maps=tex_map,              # (1, H, W, 3)
                    faces_uvs=[faces_uvs],     # list[(F,3)]
                    verts_uvs=[verts_uvs],     # list[(V,2)]
                )
                used_texture = True

        # 2) 如果没有贴图，优先顶点色（COLOR_0）
        if not used_texture and prefer_vertex_color_if_no_texture:
            vc = None
            try:
                # Trimesh 会把 glTF 的 COLOR_0 映射到 vertex_colors (Nx3 or Nx4)
                if geom.visual.vertex_colors is not None and len(geom.visual.vertex_colors) == V.shape[0]:
                    vc = np.asarray(geom.visual.vertex_colors, dtype=np.float32)
                    if vc.shape[1] == 4:
                        vc = vc[:, :3]
                    vc = vc / 255.0 if vc.max() > 1.0 else vc
            except Exception:
                vc = None

            if vc is not None:
                colors_v = vc.astype(np.float32)
                used_vertex_color = True

        # 3) 仍然没有 → 用 baseColorFactor“烘”为顶点色
        if not used_texture and not used_vertex_color:
            base_rgb = None
            try:
                mat = getattr(geom.visual, "material", None)
                if mat is not None and getattr(mat, "baseColorFactor", None) is not None:
                    # PBR 的 RGBA
                    rgba = np.array(mat.baseColorFactor, dtype=np.float32)
                    base_rgb = rgba[:3]
            except Exception:
                base_rgb = None

            if base_rgb is not None:
                colors_v[:] = base_rgb  # (R,G,B) in [0,1]

        # 收集
        verts_all.append(_to_tensor(V_w, device))
        faces_all.append(torch.from_numpy(F).to(device))
        if used_texture:
            textures_list.append(textures)
        else:
            # 用每个子网自己的顶点色
            textures_list.append(TexturesVertex(verts_features=[torch.from_numpy(colors_v).to(device)]))

    if len(verts_all) == 0:
        raise RuntimeError(f"未在 {glb_path} 中解析到可用三角网格。请确认 glb 是否包含几何。")

    # 构 Meshes 列表
    meshes = []
    for V, F, T in zip(verts_all, faces_all, textures_list):
        meshes.append(Meshes(verts=[V], faces=[F], textures=T))

    # 合成一个 scene mesh（便于一次性渲染/变换）
    scene_mesh = join_meshes_as_scene(meshes).to(device)
    return scene_mesh


def render_single_view(
    mesh: Meshes,
    image_size: int = 512,
    fov_deg: float = 60.0,
    dist: float = 2.0,
    elev_deg: float = 30.0,
    azim_deg: float = 60.0,
    device: torch.device = torch.device("cuda"),
    no_cull_backfaces: bool = False,
    brighter: bool = True,
):
    # 相机
    R, T = pytorch3d_cameras_from_spherical(dist, elev_deg, azim_deg, device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov_deg)

    # 光源（提高环境光，避免“太暗看起来全灰”）
    if brighter:
        lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]],
                            ambient_color=((0.6, 0.6, 0.6),),
                            diffuse_color=((0.7, 0.7, 0.7),),
                            specular_color=((0.2, 0.2, 0.2),))
    else:
        lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

    # 光滑栅格
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=10,
        cull_backfaces=not no_cull_backfaces,  # 默认剔背；可关闭以看内壁
        blur_radius=0.0,
        perspective_correct=True,
    )

    # 简单 Phong 材质（不是 PBR，只是给高光）
    materials = Materials(
        device=device,
        specular_color=[[0.2, 0.2, 0.2]],
        shininess=32.0
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, materials=materials),
    )

    image = renderer(mesh)  # (1, H, W, 3), float [0,1]
    rgb = (image[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return rgb


def pytorch3d_cameras_from_spherical(dist, elev_deg, azim_deg, device):
    """
    简易球坐标相机姿态（look-at 原点，+Z 朝前，+Y 向上）。
    """
    from pytorch3d.renderer import look_at_view_transform
    R, T = look_at_view_transform(dist=dist, elev=elev_deg, azim=azim_deg, device=device)
    return R, T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True, help="输入 .glb 路径")
    parser.add_argument("--out", default="glb_test.png")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--dist", type=float, default=2.0)
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--azim", type=float, default=60.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--flip_v", action="store_true", help="贴图 V 翻转（很多 glTF 需要）")
    parser.add_argument("--no_cull_backfaces", action="store_true", help="关闭背面剔除，可看内壁/中空物体内部")
    parser.add_argument("--dimmer", action="store_true", help="用较暗光照（默认更亮以避免暗灰）")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[Device] {device}")

    mesh = load_glb_as_meshes(
        args.glb,
        device=device,
        flip_v=args.flip_v,
        prefer_vertex_color_if_no_texture=True,
    )

    rgb = render_single_view(
        mesh,
        image_size=args.image_size,
        fov_deg=args.fov,
        dist=args.dist,
        elev_deg=args.elev,
        azim_deg=args.azim,
        device=device,
        no_cull_backfaces=args.no_cull_backfaces,
        brighter=not args.dimmer,
    )

    Image.fromarray(rgb).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
