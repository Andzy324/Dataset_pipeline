#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把点云颜色/标签 转移到 Mesh 顶点（或导出带顶点色的 GLB/PLY）。
- 支持：.obj / .glb / .gltf
- 可选：已知 4x4 外参；或自动尺度匹配 + ICP 对齐
- 可选：只转移某些 labels（例如把手）
- 输出：带 vertex color 的 .ply / .glb，以及 JSON 质量报告

用法示例：
python paint_pc_to_mesh.py \
  --pc points.ply \
  --mesh model.glb \
  --labels labels.npy \
  --keep keep.npy \
  --out_mesh out_colored.glb \
  --out_ply out_colored.ply \
  --report report.json \
  --estimate_transform \
  --knn 8 \
  --max-dist 0.01 \
  --lap-smooth-iters 5 \
  --lap-smooth-lambda 0.5 \
  --only-labels 1 4
"""
import argparse, json, os, sys
import numpy as np
import trimesh
import trimesh.visual
try:
    import open3d as o3d
except Exception as e:
    o3d = None
    print("[WARN] Open3D 未安装，将退化为简化功能（不支持 ICP 对齐/快速 KDTree）", file=sys.stderr)

def load_point_cloud(ply_path):
    if o3d is None:
        # 简易 PLY 读取（仅限 xyzrgb，binary_le double+uchar）
        with open(ply_path, 'rb') as f:
            header = []
            while True:
                line = f.readline()
                header.append(line.decode('latin1'))
                if b"end_header" in line: break
            data = f.read()
        dt = np.dtype([('x','<f8'),('y','<f8'),('z','<f8'),('r','u1'),('g','u1'),('b','u1')])
        arr = np.frombuffer(data, dtype=dt)
        xyz = np.stack([arr['x'],arr['y'],arr['z']], axis=1).astype(np.float64)
        rgb = np.stack([arr['r'],arr['g'],arr['b']], axis=1).astype(np.uint8)
        return xyz, rgb
    else:
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points).astype(np.float64)
        if pcd.has_colors():
            col = (np.asarray(pcd.colors)*255.0).clip(0,255).astype(np.uint8)
        else:
            col = np.full((xyz.shape[0],3), 200, np.uint8)
        return xyz, col

def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    可靠加载 .glb/.gltf/.obj：
    - 若是 Scene：对每个 geometry 应用世界变换后再 concatenate
    - 若是 Trimesh：直接返回
    - 兼容某些版本返回 list 的情况
    """
    obj = trimesh.load(mesh_path, force='scene', process=False)

    # 1) Scene 分支：按 graph 的世界变换展开每个 geometry
    if isinstance(obj, trimesh.Scene):
        meshes = []
        for name, geom in obj.geometry.items():
            # 世界变换
            try:
                T = obj.graph.get_transform(name)
            except Exception:
                T = np.eye(4)
            g = geom.copy()
            g.apply_transform(T)
            if isinstance(g, trimesh.Trimesh):
                meshes.append(g)

        if len(meshes) == 0:
            raise ValueError(f"No triangle geometry found in scene: {mesh_path}")

        mesh = trimesh.util.concatenate(meshes)

    # 2) 旧版本/特殊后端可能直接返回 list
    elif isinstance(obj, (list, tuple)):
        meshes = [m for m in obj if isinstance(m, trimesh.Trimesh)]
        if len(meshes) == 0:
            raise ValueError(f"No Trimesh in list: {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)

    # 3) 已经是 Trimesh
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj

    else:
        raise TypeError(f"Unsupported type from trimesh.load: {type(obj)}")

    # 清理一下拓扑
    try:
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_faces()
    except Exception:
        pass

    return mesh

def mesh_to_o3d(mesh: trimesh.Trimesh):
    if o3d is None:
        raise RuntimeError("需要 Open3D 执行此步骤")
    om = o3d.geometry.TriangleMesh()
    om.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    om.triangles = o3d.utility.Vector3iVector(mesh.faces)
    if mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors is not None:
        cols = np.array(mesh.visual.vertex_colors)[:,:3]/255.0
        om.vertex_colors = o3d.utility.Vector3dVector(cols)
    om.compute_vertex_normals()
    return om

def o3d_to_trimesh(om):
    v = np.asarray(om.vertices)
    f = np.asarray(om.triangles)
    col = None
    if om.has_vertex_colors():
        col = (np.asarray(om.vertex_colors)*255.0).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if col is not None:
        vc = np.zeros((v.shape[0],4), dtype=np.uint8)
        vc[:,:3] = col
        vc[:,3] = 255
        mesh.visual.vertex_colors = vc
    return mesh

def bbox_diag(xyz):
    mn, mx = xyz.min(0), xyz.max(0)
    return float(np.linalg.norm(mx - mn))

def scale_and_icp_align(pc_xyz, mesh, do_icp=True, voxel=0.005, max_iter=80):
    """ 返回：R,t,s 或 4x4 T（若使用ICP则直接返回T以便后续apply_transform） """
    # 1) 初始尺度/平移（对齐包围盒尺度 + 质心）
    def _bbox_diag(x):
        mn, mx = x.min(0), x.max(0)
        return float(np.linalg.norm(mx - mn))
    s_pc = _bbox_diag(pc_xyz)
    s_m  = _bbox_diag(mesh.vertices)
    s = 1.0 if s_m == 0 else (s_pc / s_m)
    c_pc = pc_xyz.mean(0)
    c_m  = mesh.vertices.mean(0)
    R = np.eye(3)
    t = c_pc - s * (R @ c_m)

    if not (o3d and do_icp):
        return R, t, s

    # 2) 构造 O3D 三角网格并施加初值 (S,R,t)
    om = mesh_to_o3d(mesh)  # TriangleMesh
    init_T = np.eye(4)
    init_T[:3,:3] = s * np.eye(3)
    init_T[:3, 3] = t
    om.transform(init_T)

    # 3) 采样：把 mesh → 点云（再做可选体素降采样）
    #   - 采样数量可按模型复杂度调整；过小会丢细节，过大会慢
    src_pc = om.sample_points_uniformly(number_of_points=min(60000, len(np.asarray(om.vertices))*3))
    if voxel and voxel > 0:
        src_pc = src_pc.voxel_down_sample(voxel)

    # 4) 构造目标点云
    tgt_pc = o3d.geometry.PointCloud()
    tgt_pc.points = o3d.utility.Vector3dVector(pc_xyz)
    if voxel and voxel > 0:
        tgt_pc = tgt_pc.voxel_down_sample(voxel)
    
    # 5) ICP（点到点）
    max_corr = voxel*5 if voxel else 0.02  # 可按规模调
    
    reg = o3d.pipelines.registration.registration_icp(
        source=src_pc,
        target=tgt_pc,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    # reg.transformation 是在 om 已应用 init_T 的基础上进一步配准的增量
    T_refine = reg.transformation
    T_total = T_refine @ init_T
    return None, None, None, T_total

def apply_transform_to_mesh(mesh, R=None, t=None, s=None, T4=None):
    if T4 is not None:
        mesh.apply_transform(T4)
    else:
        S = np.eye(4); S[:3,:3] *= (s if s is not None else 1.0)
        RT = np.eye(4); RT[:3,:3] = (R if R is not None else np.eye(3)); RT[:3,3] = (t if t is not None else 0.0)
        mesh.apply_transform(S @ RT)

# def knn_color_transfer(pc_xyz, pc_rgb, mesh, knn=8, max_dist=0.01, only_labels=None, labels=None):
#     # 1) 过滤标签（如只上色把手）
#     use_xyz, use_rgb = pc_xyz, pc_rgb
#     if only_labels is not None and labels is not None:
#         m = np.isin(labels, np.array(only_labels, dtype=labels.dtype))
#         use_xyz, use_rgb = pc_xyz[m], pc_rgb[m]
#         if use_xyz.shape[0] == 0:
#             raise ValueError("筛选标签后点云为空。")

#     # 2) 保证 dtype & 有效值
#     use_xyz = np.ascontiguousarray(use_xyz, dtype=np.float64)
#     finite_mask = np.isfinite(use_xyz).all(axis=1)
#     if not finite_mask.all():
#         use_xyz = use_xyz[finite_mask]
#         use_rgb = use_rgb[finite_mask]
#     if use_xyz.shape[0] == 0:
#         raise ValueError("点云为空或全为非有限值。")

#     # 3) 构造 KDTree（Open3D 推荐：基于 PointCloud）
#     def _build_kdtree(xyz):
#         if o3d is None:
#             return None, None
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)  # float64
#         kdt = o3d.geometry.KDTreeFlann(pcd)
#         return kdt, pcd

#     kdt, _pcd = _build_kdtree(use_xyz)

#     # 4) 查询函数（含安全保护与降级）
#     Np = use_xyz.shape[0]
#     def query(p, k):
#         safe_k = int(min(max(k, 1), Np))
#         p = np.asarray(p, dtype=np.float64).ravel()
#         if p.shape[0] != 3 or not np.isfinite(p).all() or safe_k < 1:
#             return np.array([], dtype=int), np.array([], dtype=float)

#         if kdt is not None:
#             try:
#                 ok, idx, dist2 = kdt.search_knn_vector_3d(p, safe_k)
#                 if ok == 0:
#                     return np.array([], dtype=int), np.array([], dtype=float)
#                 # Open3D 返回的 idx 是 python list/array，dist2 是 list
#                 idx = np.asarray(idx, dtype=int)
#                 dist2 = np.asarray(dist2, dtype=float)
#                 return idx, dist2
#             except Exception:
#                 pass  # 降级到暴力

#         # 暴力降级（小数据可用）
#         d2 = np.sum((use_xyz - p) ** 2, axis=1)
#         idx = np.argpartition(d2, safe_k - 1)[:safe_k]
#         return idx, d2[idx]

#     # 5) 顶点着色
#     v = np.asarray(mesh.vertices, dtype=np.float64)
#     out = np.zeros((v.shape[0], 3), np.uint8)
#     dists = np.full(v.shape[0], np.nan, float)

#     for i, p in enumerate(v):
#         idx, d2 = query(p, knn)
#         if idx.size == 0:
#             out[i] = [128, 128, 128]
#             continue

#         d = np.sqrt(d2)
#         dists[i] = float(np.min(d))
#         # 距离阈值过滤
#         if max_dist is not None:
#             mask = d <= max_dist
#             if not np.any(mask):
#                 out[i] = [128, 128, 128]
#                 continue
#             idx, d = idx[mask], d[mask]

#         # 距离加权平均
#         w = 1.0 / (d + 1e-8)
#         w = w / w.sum()
#         col = (use_rgb[idx].astype(np.float32) * w[:, None]).sum(0)
#         out[i] = col.clip(0, 255).astype(np.uint8)

#     metrics = {
#         "coverage_ratio": float(np.isfinite(dists).mean()),
#         "mean_nn_distance": float(np.nanmean(dists)),
#         "median_nn_distance": float(np.nanmedian(dists)),
#         "over_threshold_ratio": float(np.mean(dists > (max_dist if max_dist else np.inf))),
#         "used_backend": "open3d" if kdt is not None else "bruteforce"
#     }
#     return out, metrics
def knn_color_transfer(
    pc_xyz, pc_rgb, mesh,
    knn=8, max_dist=0.03,
    only_labels=None, labels=None,
    min_neighbors=3, knn_sigma_scale=0.5,
    knn_percentile=90.0, use_normal_weight=False
):
    """
    将点云颜色以 kNN 投到 mesh 顶点：
    - 高斯距离权重 + 可选法线权重 + 鲁棒截断
    - 阈值过滤 + 最少邻居数
    - 支持 only_labels 指定只上色某些标签
    返回：vertex_rgb(uint8 Nx3), metrics(dict)
    """
    # 1) 选择点（可选标签过滤）
    use_xyz, use_rgb = pc_xyz, pc_rgb
    if only_labels is not None and labels is not None:
        m = np.isin(labels, np.asarray(only_labels, dtype=labels.dtype))
        use_xyz, use_rgb = pc_xyz[m], pc_rgb[m]
    if use_xyz.size == 0:
        raise ValueError("knn_color_transfer: 用于着色的点云为空。")

    # 2) 数据清洁 & dtype
    use_xyz = np.ascontiguousarray(use_xyz, dtype=np.float64)
    good = np.isfinite(use_xyz).all(axis=1)
    if not np.all(good):
        use_xyz, use_rgb = use_xyz[good], use_rgb[good]
    Np = use_xyz.shape[0]
    if Np == 0:
        raise ValueError("knn_color_transfer: 点云坐标全为非有限值。")

    # 3) KDTree（Open3D 优先）
    kdt, pcd = None, None
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(use_xyz)
        kdt = o3d.geometry.KDTreeFlann(pcd)

    # 4) 查询函数（失败自动暴力降级）
    def query_knn(p, k):
        k_eff = int(min(max(k, 1), Np))
        p = np.asarray(p, dtype=np.float64).ravel()
        if p.shape[0] != 3 or not np.isfinite(p).all():
            return np.empty(0, int), np.empty(0, float)
        if kdt is not None:
            try:
                ok, idx, dist2 = kdt.search_knn_vector_3d(p, k_eff)
                if ok == 0:
                    return np.empty(0, int), np.empty(0, float)
                return np.asarray(idx, int), np.asarray(dist2, float)
            except Exception:
                pass
        # 暴力降级
        d2 = np.sum((use_xyz - p)**2, axis=1)
        idx = np.argpartition(d2, k_eff-1)[:k_eff]
        return idx, d2[idx]

    # 5) 预备法线（可选用于角度权重）
    mesh_vert = np.asarray(mesh.vertices, dtype=np.float64)
    have_normals = hasattr(mesh, 'vertex_normals') and \
                   mesh.vertex_normals is not None and \
                   len(mesh.vertex_normals) == len(mesh_vert)
    if use_normal_weight and not have_normals:
        try:
            mesh.vertex_normals  # 若没有，尝试让库自动生成
        except Exception:
            pass
        have_normals = hasattr(mesh, 'vertex_normals') and \
                       mesh.vertex_normals is not None and \
                       len(mesh.vertex_normals) == len(mesh_vert)

    out = np.zeros((mesh_vert.shape[0], 3), np.uint8)
    dists = np.full(mesh_vert.shape[0], np.nan, float)

    # 6) 主循环：对每个顶点做 knn 融合
    for i, p in enumerate(mesh_vert):
        idx, d2 = query_knn(p, knn)
        if idx.size == 0:
            out[i] = (128,128,128)
            continue

        d = np.sqrt(d2)
        dists[i] = float(np.min(d)) if d.size else np.nan

        # 距离阈值过滤
        mask = np.ones_like(d, dtype=bool)
        if max_dist is not None:
            mask &= (d <= max_dist)

        # 鲁棒截断（去掉最远的若干邻居）
        if 0.0 < knn_percentile < 100.0 and d.size > 0:
            cutoff = np.percentile(d, knn_percentile)
            mask &= (d <= cutoff)

        if not np.any(mask):
            out[i] = (128,128,128); continue

        idx, d = idx[mask], d[mask]
        if idx.size < max(1, min_neighbors):
            # 邻居过少，保护：可选择置灰或放宽到最近的 min_neighbors 个
            take = min(len(d), min_neighbors)
            sel = np.argsort(d)[:take]
            idx, d = idx[sel], d[sel]

        # 高斯距离权重
        if max_dist is not None and max_dist > 0:
            sigma = max_dist * float(knn_sigma_scale)
        else:
            sigma = float(np.median(d) + 1e-6)
        w_d = np.exp(-(d**2) / (2.0 * sigma**2))

        # 法线/朝向权重（可选）：鼓励“同侧/正对”的邻居
        if use_normal_weight and have_normals:
            n = np.asarray(mesh.vertex_normals, dtype=np.float64)[i]
            v = use_xyz[idx] - p
            v_norm = np.linalg.norm(v, axis=1) + 1e-12
            cos = np.clip((v @ n) / v_norm, -1.0, 1.0)
            w_ang = np.maximum(0.0, cos)**2  # 指数可调
            w = w_d * w_ang
        else:
            w = w_d

        if not np.isfinite(w).any() or w.sum() <= 1e-12:
            out[i] = (128,128,128); continue

        w = w / w.sum()
        col = (use_rgb[idx].astype(np.float32) * w[:,None]).sum(0)
        out[i] = col.clip(0,255).astype(np.uint8)

    metrics = {
        "coverage_ratio": float(np.isfinite(dists).mean()),
        "mean_nn_distance": float(np.nanmean(dists)),
        "median_nn_distance": float(np.nanmedian(dists)),
        "over_threshold_ratio": float(np.mean(dists > (max_dist if max_dist else np.inf))),
        "used_backend": "open3d" if kdt is not None else "bruteforce"
    }
    return out, metrics

def laplacian_smooth_colors(mesh, colors, iters=0, lamb=0.5):
    if iters<=0: return colors
    # 顶点邻接平均做简易颜色平滑
    adj = mesh.vertex_adjacency_graph.to_coo()
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones_like(adj.data), (adj.row, adj.col)), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0]))
    Dinv = sp.diags(1.0 / np.maximum(1, np.array(A.sum(1)).ravel()))
    W = Dinv @ A
    C = colors.astype(np.float32)
    for _ in range(iters):
        C = (1-lamb)*C + lamb*(W @ C)
    return C.clip(0,255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc', required=True)
    ap.add_argument('--mesh', required=True)
    ap.add_argument('--labels', default=None)
    ap.add_argument('--keep', default=None)
    ap.add_argument('--only_labels', type=int, nargs='*', default=None, help='只转移这些label（可多选）')
    ap.add_argument('--known_transform', default=None, help='txt/npz：4x4 外参(世界←点云/模型)')
    ap.add_argument('--estimate_transform', dest='estimate_transform', action='store_true',
                help='自动尺度匹配 + ICP 对齐')
    ap.add_argument('--pc_rot_preset',
                choices=['none','zup_to_yup','yup_to_zup'],
                default='none',
                help='预对齐：Z-up<->Y-up（右手系）')
    ap.add_argument('--rot_pivot',
                choices=['pc_centroid','mesh_centroid','origin'],
                default='pc_centroid',
                help='预旋的旋转中心（枢轴）')
    ap.add_argument('--pc_rot_auto_yz', action='store_true',
                help='自动在 zup_to_yup / yup_to_zup 中择优')
    ap.add_argument('--knn', type=int, default=8)
    ap.add_argument('--max_dist', type=float, default=0.01)
    ap.add_argument('--lap_smooth_iters', type=int, default=0)
    ap.add_argument('--lap-smooth_lambda', type=float, default=0.5)
    ap.add_argument('--out_mesh', default=None, help='导出 glb/gltf/obj，含顶点色')
    ap.add_argument('--out_ply',  default=None, help='导出 ply，含顶点色')
    ap.add_argument('--report',   default=None, help='质量报告 json')
    args = ap.parse_args()

    pc_xyz, pc_rgb = load_point_cloud(args.pc)
    
    def apply_pre_rotation(pc_xyz, preset, pivot='pc_centroid', mesh=None):
        if preset in (None, 'none'):
            return pc_xyz
        if preset == 'zup_to_yup':
            # Rx(+90°): Y->-Z, Z->Y
            R = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
        elif preset == 'yup_to_zup':
            # Rx(-90°): Y->Z, Z->-Y
            R = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
        else:
            raise ValueError("bad preset")

        if pivot == 'pc_centroid':
            c = pc_xyz.mean(0)
        elif pivot == 'mesh_centroid' and mesh is not None:
            c = mesh.vertices.mean(0)
        else:
            c = np.zeros(3)

        return (pc_xyz - c) @ R.T + c


    # pc_xyz = apply_pre_rotation(pc_xyz, args.pc_rot_preset, args.rot_pivot, mesh=None)  # mesh 可在加载后再传
    labels = np.load(args.labels) if args.labels else None
    if args.keep:
        keep = np.load(args.keep)
        # 如需与原索引系联动，可在此使用 keep；对当前 50k 子集直接着色，不强制依赖 keep
    mesh = load_mesh(args.mesh)

    # 已知外参 or 自动对齐
    if args.known_transform:
        T = np.loadtxt(args.known_transform) if args.known_transform.endswith('.txt') else np.load(args.known_transform)['T']
        mesh.apply_transform(T)
    elif args.estimate_transform:
        if o3d is None:
            print("[WARN] 未安装 Open3D，无法自动 ICP，对齐将仅做尺度与中心匹配。", file=sys.stderr)
            R,t,s = scale_and_icp_align(pc_xyz, mesh, do_icp=False)
            apply_transform_to_mesh(mesh, R,t,s)
        else:
            res = scale_and_icp_align(pc_xyz, mesh, do_icp=True)
            if len(res)==4: _,_,_,T = res; apply_transform_to_mesh(mesh, T4=T)
            else: R,t,s,_ = (*res, None); apply_transform_to_mesh(mesh, R,t,s)

    Rx_plus_90 = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
    T_yup = np.eye(4); T_yup[:3,:3] = Rx_plus_90
    mesh.apply_transform(T_yup)
    # 颜色转移
    vertex_rgb, metrics = knn_color_transfer(
        pc_xyz, pc_rgb, mesh,
        knn=args.knn, max_dist=args.max_dist,
        only_labels=args.only_labels, labels=labels
    )
    if args.lap_smooth_iters>0:
        try:
            vertex_rgb = laplacian_smooth_colors(mesh, vertex_rgb, args.lap_smooth_iters, args.lap_smooth_lambda)
        except Exception as e:
            print("[WARN] 颜色平滑失败（缺少 scipy ？），将跳过。", file=sys.stderr)

    # 写回 mesh 顶点色并导
    N = mesh.vertices.shape[0]
        # —— 回传出顶点色 vertex_rgb 之后，强制切到“仅顶点色”的视觉：
    vc = np.zeros((mesh.vertices.shape[0],4), np.uint8)
    vc[:,:3] = vertex_rgb
    vc[:,3]  = 255

    # 关键：彻底移除材质/贴图/UV，避免 glb 里还带旧贴图
    try:
        import trimesh
        from trimesh.visual import ColorVisuals
        # 清掉 UV 和材质引用（有些 viewer 会把 BaseColorTexture 覆盖顶点色视觉）
        if hasattr(mesh.visual, 'uv'):       mesh.visual.uv = None
        if hasattr(mesh.visual, 'material'): mesh.visual.material = None
        mesh.visual = ColorVisuals(mesh, vertex_colors=vc)  # 只留顶点色
    except Exception as e:
        mesh.visual.vertex_colors = vc  # 退化兜底

    # —— 导出：用 glTF 导出器显式写 COLOR_0
    if args.out_mesh and args.out_mesh.lower().endswith('.glb'):
        from trimesh.exchange import gltf
        glb_bytes = gltf.export_glb(mesh, include_normals=True)
        with open(args.out_mesh, 'wb') as f:
            f.write(glb_bytes)
        print(f"[OK] 导出带顶点色 GLB: {args.out_mesh}")
    else:
        # 其它格式或未指定 .glb 时，用通用导出（也会带顶点色，但优先建议上面的 glb 分支）
        if args.out_mesh:
            mesh.export(args.out_mesh)
            print(f"[OK] 导出 Mesh: {args.out_mesh}")
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] 质量报告: {args.report}\n{json.dumps(metrics, indent=2)}")     

if __name__ == "__main__":
    main()
