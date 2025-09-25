#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pc2asset_pipeline.py
一键完成：
  1) headless Blender 对 GLB 恢复性旋转 + 对齐报告 (JSON)
  2) 点云→网格 颜色回传（kNN，支持只上色选定 labels；可选 ICP 微调）
  3) headless Blender 烘焙 顶点色→纹理，导出 OBJ + MTL + textures/

示例：
python pc2asset_pipeline.py \
  --pc ./points.ply \
  --glb ./mug_965_4831669.glb \
  --rx_deg 90 \
  --pivot pc_centroid \
  --knn 12 --max_dist 0.04 --icp \
  --workdir ./_out \
  --make_obj

作者提示：
- 若你的点云是 Z-up，而 GLB 在 Blender 里看被 -90° 旋转，可设 --rx_deg 90 作为“恢复性旋转”，
  让 GLB 烘焙回与点云一致的 up 轴（建议以点云为工作坐标系）。
- 导出的 GLB 使用 COLOR_0 顶点色；OBJ 使用烘焙出的纹理（OBJ 对顶点色兼容性差）。
"""
import os, sys, json, tempfile, subprocess, textwrap, argparse
import numpy as np, trimesh

try:
    import open3d as o3d
except Exception as e:
    o3d = None

# ----------------------------- Blender 子脚本 -----------------------------

BLENDER_ALIGN_PY = r"""
# --- REPLACE WHOLE FILE CONTENT OF blender_align.py WITH THIS ---
import bpy, sys, json, math, os
from mathutils import Matrix, Vector

argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
def pop(flag, default=None, cast=str):
    if flag in argv:
        i = argv.index(flag); v = cast(argv[i+1]); del argv[i:i+2]; return v
    return default

IN_GLB   = pop("--in_glb")
OUT_GLB  = pop("--out_glb", "aligned.glb")
REPORT   = pop("--report",  "align_report.json")
RX_DEG   = pop("--rx_deg",  0.0, float)
PIVOT    = pop("--pivot", "pc_centroid")   # mesh_centroid|pc_centroid|origin
PIVOT_XYZ= pop("--pivot_xyz", None)        # "x,y,z"（优先生效）

if not (IN_GLB and os.path.exists(IN_GLB)):
    raise RuntimeError(f"Missing --in_glb: {IN_GLB}")
out_dir = os.path.dirname(OUT_GLB)
if out_dir: os.makedirs(out_dir, exist_ok=True)

bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.gltf(filepath=IN_GLB)

meshes = [o for o in bpy.context.scene.objects if o.type=='MESH']
if not meshes: raise RuntimeError("No mesh objects in GLB.")
bpy.ops.object.select_all(action='DESELECT')
for o in meshes: o.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes)>1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

def evaluated(o):
    return o.evaluated_get(bpy.context.evaluated_depsgraph_get())

def bbox_center_world(o):
    e = evaluated(o)
    corners = [e.matrix_world @ Vector(c) for c in e.bound_box]
    c = Vector((0,0,0))
    for v in corners: c += v
    return c/8.0

def geom_centroid_world(o):
    e = evaluated(o)
    deps = bpy.context.evaluated_depsgraph_get()
    me = e.to_mesh(preserve_all_data_layers=False, depsgraph=deps)
    try:
        n = len(me.vertices)
        if n==0: return bbox_center_world(o)
        acc = Vector((0,0,0)); mw = e.matrix_world
        for v in me.vertices: acc += (mw @ v.co)
        return acc / n
    finally:
        e.to_mesh_clear()

def make_report(prefix="mesh"):
    c  = geom_centroid_world(obj)
    bc = bbox_center_world(obj)
    loc= evaluated(obj).matrix_world.translation
    return {
        f"{prefix}_location": [loc.x,loc.y,loc.z],
        f"{prefix}_centroid": [c.x,c.y,c.z],
        f"{prefix}_bbox_center": [bc.x,bc.y,bc.z]
    }

rep = {}
rep.update(make_report("mesh_before"))

# pivot 选择
if PIVOT_XYZ:
    x,y,z = (float(t) for t in PIVOT_XYZ.split(","))
    pivot = Vector((x,y,z))
elif PIVOT=="mesh_centroid":
    pivot = geom_centroid_world(obj)
elif PIVOT=="origin":
    pivot = Vector((0,0,0))
else:
    # 未传 pivot_xyz 时，退化为网格质心
    pivot = geom_centroid_world(obj)

# 恢复性旋转（绕 X），烘焙进几何
if abs(RX_DEG) > 1e-7:
    Rx = Matrix.Rotation(math.radians(RX_DEG), 4, 'X')
    T  = Matrix.Translation(pivot) @ Rx @ Matrix.Translation(-pivot)
    obj.matrix_world = T @ obj.matrix_world
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

bpy.context.view_layer.update()
rep.update(make_report("mesh_after"))
with open(REPORT, "w", encoding="utf-8") as f: json.dump(rep, f, indent=2)
print("[align] report saved:", REPORT)

bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format='GLB', use_selection=True)
print("[align] aligned GLB saved:", OUT_GLB)

# --- END OF FILE ---

# pc_centroid = None
# if IN_PLY and os.path.exists(IN_PLY):
#     bpy.ops.import_mesh.ply(filepath=IN_PLY)
#     plys = [o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.lower().endswith(".ply")]
#     if plys:
#         pc = plys[-1]
#         pc_centroid = centroid(pc)
#         rep["pc_centroid"] = list(pc_centroid)
#         pc.hide_viewport = True; pc.hide_render = True

# pivot
# if PIVOT=="pc_centroid" and pc_centroid:
#     pivot = pc_centroid
# elif PIVOT=="mesh_centroid":
#     pivot = centroid(obj)
# else:
#     pivot = Vector((0,0,0))

# # 恢复性旋转（绕X）
# if abs(RX_DEG) > 1e-7:
#     Rx = Matrix.Rotation(math.radians(RX_DEG), 4, 'X')
#     T = Matrix.Translation(pivot) @ Rx @ Matrix.Translation(-pivot)
#     obj.matrix_world = T @ obj.matrix_world
#     bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

# rep["mesh_after"] = {
#     "location": list(obj.matrix_world.translation),
#     "centroid": list(centroid(obj)),
#     "bbox_center": list(bbox_center(obj))
# }
# with open(REPORT, "w") as f: json.dump(rep, f, indent=2)
# print("[align] report saved:", REPORT)

# # 导出（几何已烘焙）
# bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True)
# bpy.ops.export_scene.gltf(filepath=OUT_GLB, export_format='GLB', use_selection=True)
# print("[align] aligned GLB saved:", OUT_GLB)
"""

BLENDER_BAKE_PY = r"""
# bake_vcol_to_obj.py — robust bake of vertex colors to PNG, then export OBJ+MTL+textures
# Blender 4.5+ compatible; headless OK.

import bpy, sys, os, bmesh

# ---- args ----
argv = sys.argv
argv = argv[argv.index("--")+1:] if "--" in argv else []
def pop(flag, default=None, cast=str):
    if flag in argv:
        i = argv.index(flag); v = cast(argv[i+1]); del argv[i:i+2]; return v
    return default

IN_MESH    = pop("--in_mesh")
OUT_DIR    = pop("--out_dir", "./_obj")
TEX_NAME   = pop("--tex_name", "albedo")
TEX_RES    = pop("--tex_res", 2048, int)
ATTR_NAME  = pop("--attr_name", None)           # 顶点色层名，如 COLOR_0
BASE_RGB   = pop("--base_rgb", "200,200,200")   # 未上色区域底色
V_EPS      = pop("--vcol_threshold", "0.02")    # 顶点色“近黑”阈值(线性RGB)
UV_MARGIN  = pop("--uv_margin", 0.02, float)
PER_FACE   = pop("--per_face_islands", "true")  # 是否每面一个岛（true/false）
CSEAM_THR  = pop("--color_seam_thresh", "0.08") # 按色差打 seam 的阈值

if not (IN_MESH and os.path.exists(IN_MESH)):
    raise RuntimeError(f"Missing --in_mesh or not found: {IN_MESH}")
OUT_DIR = os.path.join(OUT_DIR, "_obj") 
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "textures"), exist_ok=True)
tex_path = os.path.join(OUT_DIR, "textures", f"{TEX_NAME}.png")

br, bg, bb = [max(0, min(255, int(x))) for x in BASE_RGB.split(",")]
base_lin = (br/255.0, bg/255.0, bb/255.0, 1.0)
v_eps = float(V_EPS)
per_face_islands = (str(PER_FACE).lower()=="true")
cseam_thr = float(CSEAM_THR)

# ---- scene ----
bpy.ops.wm.read_homefile(use_empty=True)
bpy.ops.import_scene.gltf(filepath=IN_MESH)

objs = [o for o in bpy.context.scene.objects if o.type=='MESH']
if not objs: raise RuntimeError("No mesh in GLB.")
bpy.ops.object.select_all(action='DESELECT')
for o in objs: o.select_set(True)
bpy.context.view_layer.objects.active = objs[0]
if len(objs)>1: bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active

# ---- pick vertex color layer ----
def list_color_layers(me):
    ca = getattr(me, "color_attributes", None)
    layers = []
    if ca:
        for a in ca:
            s=0.0; n=min(200, len(a.data))
            for i in range(n): 
                c=a.data[i].color; s+=(c[0]+c[1]+c[2])
            layers.append((a.name, a.domain, a.data_type, s>1e-6))
    else:
        vc = getattr(me, "vertex_colors", None)
        if vc and len(vc)>0:
            a = vc.active or vc[0]
            layers.append((a.name, 'CORNER', 'BYTE_COLOR', True))
    return layers

def pick_layer(me, prefer=None):
    layers = list_color_layers(me)
    if not layers: return None
    def exists(n): return any(n==x[0] for x in layers)
    order=[]
    if prefer and exists(prefer): order.append(prefer)
    if exists("COLOR_0"): order.append("COLOR_0")
    order.append(layers[0][0])
    for n,_,_,ok in layers:
        if ok and n not in order: order.append(n)
    for cand in order:
        for n,_,_,ok in layers:
            if n==cand and ok: return cand
    return layers[0][0]

layer = pick_layer(obj.data, ATTR_NAME)
if not layer: raise RuntimeError("No vertex color layer.")
print("[bake] Using vcol layer:", layer)

# ---- edit mesh in bmesh: fill near-black to base, mark seams ----
bm = bmesh.new(); bm.from_mesh(obj.data)
uv_layer = bm.loops.layers.uv.verify()
vcol_layer = bm.loops.layers.color.get(layer)
if vcol_layer is None:
    vcol_layer = bm.loops.layers.color.new(layer)
    for f in bm.faces:
        for l in f.loops: l[vcol_layer]=base_lin

# 1) 近黑改底色（数据层面，不用 shader 混合）
for f in bm.faces:
    for l in f.loops:
        r,g,b,_ = l[vcol_layer]
        if (r+g+b) < v_eps: l[vcol_layer] = base_lin

# 2) 按需求打 seam
for e in bm.edges: e.seam=False
if per_face_islands:
    for e in bm.edges: e.seam=True
else:
    # 按面平均色差打 seam
    def favg(fc):
        r=g=b=0.0
        for lp in fc.loops:
            c=lp[vcol_layer]; r+=c[0]; g+=c[1]; b+=c[2]
        n=len(fc.loops); return (r/n,g/n,b/n)
    for e in bm.edges:
        if len(e.link_faces)!=2: 
            e.seam=True; continue
        c1=favg(e.link_faces[0]); c2=favg(e.link_faces[1])
        diff=abs(c1[0]-c2[0])+abs(c1[1]-c2[1])+abs(c1[2]-c2[2])
        if diff>=cseam_thr: e.seam=True

bm.to_mesh(obj.data); bm.free(); bpy.context.view_layer.update()

# ---- unwrap by seams（硬边） ----
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=UV_MARGIN, use_subsurf_data=False)
bpy.ops.uv.pack_islands(margin=UV_MARGIN)
bpy.ops.object.mode_set(mode='OBJECT')

# ---- build emission material (no mixing) ----
mat = bpy.data.materials.new(name="BakeFromVCol")
mat.use_nodes=True
nt=mat.node_tree; nodes=nt.nodes; links=nt.links
for n in list(nodes): nodes.remove(n)
out = nodes.new("ShaderNodeOutputMaterial"); out.location=(300,0)
em  = nodes.new("ShaderNodeEmission");        em.location=(60,0)
img = bpy.data.images.new(name=TEX_NAME, width=TEX_RES, height=TEX_RES, alpha=False)
img.colorspace_settings.name='sRGB'
img.pixels = list(base_lin) * (TEX_RES*TEX_RES)  # 预填底色
imgtex = nodes.new("ShaderNodeTexImage"); imgtex.location=(60,-200); imgtex.image=img; imgtex.interpolation='Closest'

try:
    vcol = nodes.new("ShaderNodeVertexColor"); vcol.location=(-140,0); vcol.layer_name=layer
    vout = vcol.outputs.get("Color") or vcol.outputs[0]
except Exception:
    vcol = nodes.new("ShaderNodeAttribute"); vcol.location=(-140,0); vcol.attribute_name=layer
    vout = vcol.outputs.get("Color") or vcol.outputs[0]

links.new(vout, em.inputs.get("Color") or em.inputs[0])
links.new(em.outputs.get("Emission") or em.outputs[0], out.inputs.get("Surface") or out.inputs[0])
nt.nodes.active = imgtex

obj.data.materials.clear(); obj.data.materials.append(mat)

# ---- bake emit ----
bpy.context.scene.render.engine='CYCLES'
bpy.context.scene.cycles.device='CPU'
bpy.context.scene.cycles.samples=1
bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True)
bpy.context.view_layer.objects.active=obj
bpy.ops.object.bake(type='EMIT', margin=8)
img.filepath_raw=tex_path; img.file_format='PNG'; img.save()
print("[bake] Texture saved:", tex_path)

# ---- simple PBR with baked texture ----
std=bpy.data.materials.new(name="Textured"); std.use_nodes=True
nt2=std.node_tree; ns2=nt2.nodes; ls2=nt2.links
for n in list(ns2): ns2.remove(n)
out2=ns2.new("ShaderNodeOutputMaterial"); out2.location=(300,0)
bsdf=ns2.new("ShaderNodeBsdfPrincipled"); bsdf.location=(60,0)
tex =ns2.new("ShaderNodeTexImage");       tex.location=(-140,0)
tex.image=bpy.data.images.load(tex_path, check_existing=True)
ls2.new(tex.outputs.get("Color") or tex.outputs[0], bsdf.inputs.get("Base Color") or bsdf.inputs[0])
ls2.new(bsdf.outputs.get("BSDF") or bsdf.outputs[0], out2.inputs.get("Surface") or out2.inputs[0])
obj.data.materials.clear(); obj.data.materials.append(std)

# ---- unify normals ----
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

# ---- export OBJ ----
out_obj = os.path.join(OUT_DIR, "model.obj")
ok=False
if hasattr(bpy.ops.export_scene,"obj"):
    try:
        bpy.ops.export_scene.obj(
            filepath=out_obj, use_materials=True, path_mode='RELATIVE',
            keep_vertex_order=True, axis_forward='-Z', axis_up='Y'
        ); ok=True
    except Exception as e:
        print("[bake] export_scene.obj failed, try wm.obj_export:", e)
if not ok:
    bpy.ops.wm.obj_export(
        filepath=out_obj, export_materials=True, path_mode='RELATIVE',
        forward_axis='NEGATIVE_Z', up_axis='Y'
    )
print("[bake] OBJ exported:", out_obj)
print("[bake] Done.")
"""


# ----------------------------- Python 实现段 -----------------------------

def sh(cmd, check=True):
    print("[run]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return p

def write_tmp_script(content, name, tmpdir):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content))
    return path

def compute_metrics_for_vertices(pc_xyz, V, colored_mask, max_dist=0.01):
    """
    基于点云最近邻，计算网格顶点的覆盖与距离统计：
    - colored_mask: 哪些顶点被认为“应着色”（True 的会参与统计；False 的记为 NaN）
    返回: metrics(dict)
    """
    assert o3d is not None, "open3d is required for metrics"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pc_xyz, dtype=np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    V = np.asarray(V, dtype=np.float64)
    dmins = np.full(V.shape[0], np.nan, float)

    for i, p in enumerate(V):
        if not colored_mask[i]:
            continue
        ok, ids, d2 = kdt.search_knn_vector_3d(p, 1)
        if ok == 0:
            continue
        dmins[i] = np.sqrt(d2[0])

    # 统计（只对有限值）
    finite = np.isfinite(dmins)
    metrics = {
        "coverage_ratio": float(finite.mean()) if finite.size else 0.0,
        "mean_nn_distance": float(np.nanmean(dmins)) if finite.any() else float("nan"),
        "median_nn_distance": float(np.nanmedian(dmins)) if finite.any() else float("nan"),
        "over_threshold_ratio": float(np.mean(dmins > max_dist)) if finite.any() else 1.0,
        "used_backend": "open3d"
    }
    return metrics

def load_point_cloud(ply_path):
    if o3d is None:
        raise RuntimeError("open3d not installed. pip install open3d")
    pcd = o3d.io.read_point_cloud(ply_path)
    xyz = np.asarray(pcd.points).astype(np.float64)
    if pcd.has_colors():
        rgb = (np.asarray(pcd.colors)*255).clip(0,255).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0],3), 200, np.uint8)
    return xyz, rgb

def load_mesh(mesh_path):
    m = trimesh.load(mesh_path, force='mesh')
    # 替代已弃用API
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    return m

def icp_refine(mesh, pc_xyz, voxel=0.005, max_iter=80):
    """mesh(Trimesh)→PointCloud采样后与pc做点到点ICP，返回 4x4 T"""
    om = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )
    init_T = np.eye(4)
    # 采样
    src = om.sample_points_uniformly(number_of_points=min(80000, len(mesh.vertices)*3))
    if voxel and voxel>0: src = src.voxel_down_sample(voxel)
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_xyz))
    if voxel and voxel>0: tgt = tgt.voxel_down_sample(voxel)
    max_corr = voxel*5 if voxel else 0.02
    reg = o3d.pipelines.registration.registration_icp(
        source=src, target=tgt, max_correspondence_distance=max_corr, init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return reg.transformation

# def knn_transfer(pc_xyz, pc_rgb, mesh, knn=8, max_dist=0.03, labels=None, only_labels=None):
#     # 过滤标签（可选）
#     use_xyz, use_rgb = pc_xyz, pc_rgb
#     if labels is not None and only_labels:
#         m = np.isin(labels, np.array(only_labels, dtype=labels.dtype))
#         use_xyz, use_rgb = pc_xyz[m], pc_rgb[m]
#         if use_xyz.shape[0]==0:
#             raise ValueError("筛选标签后点云为空。")

#     # KDTree
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(use_xyz.astype(np.float64))
#     kdt = o3d.geometry.KDTreeFlann(pcd)
#     V = mesh.vertices
#     out = np.zeros((V.shape[0],3), np.uint8)
#     dists = np.full(V.shape[0], np.nan, float)

#     for i, p in enumerate(V):
#         ok, idx, d2 = kdt.search_knn_vector_3d(p.astype(np.float64), min(knn, len(use_xyz)))
#         if ok==0:
#             out[i] = (128,128,128); continue
#         d = np.sqrt(np.asarray(d2))
#         dists[i] = float(d.min())
#         if max_dist is not None:
#             mask = d <= max_dist
#             if not np.any(mask):
#                 out[i] = (128,128,128); continue
#             idx = np.asarray(idx)[mask]; d = d[mask]
#         w = 1.0/(d+1e-8); w/=w.sum()
#         col = (use_rgb[idx].astype(np.float32)*w[:,None]).sum(0)
#         out[i] = col.clip(0,255).astype(np.uint8)

#     metrics = {
#         "coverage_ratio": float(np.isfinite(dists).mean()),
#         "mean_nn_distance": float(np.nanmean(dists)),
#         "median_nn_distance": float(np.nanmedian(dists)),
#         "over_threshold_ratio": float(np.mean(dists > (max_dist if max_dist else np.inf)))
#     }
#     return out, metrics
def color_transfer(pc_xyz, pc_rgb, mesh,
                   neighbor_mode='knn', knn=8, radius=None,
                   max_dist=0.01,
                   labels=None, only_labels=None,
                   trim_frac=0.3, min_inlier_frac=0.6,
                   normal_consistency_deg=None):
    """
    将点云颜色转移到 mesh 顶点，支持：
    - 仅用指定 labels 的点（only_labels）
    - 邻域模式：kNN 或 radius
    - 距离硬阈值 max_dist
    - 修剪 trim_frac：丢弃邻域中最远的一部分
    - 语义一致性：邻域中允许 labels 的比例不足时不着色
    - 可选法向一致性：要求 mesh 顶点法向与点云法向夹角 <= 阈值

    返回：vertex_rgb(uint8 Nx3), metrics(dict)
    """
    import numpy as np
    if o4 := o3d is None:
        raise RuntimeError("open3d 未安装")

    # 1) 过滤允许的标签
    use_xyz, use_rgb, use_lab = pc_xyz, pc_rgb, labels
    if only_labels is not None and labels is not None:
        m = np.isin(labels, np.array(only_labels, dtype=labels.dtype))
        use_xyz, use_rgb = pc_xyz[m], pc_rgb[m]
        use_lab = labels[m]
        if use_xyz.shape[0] == 0:
            raise ValueError("筛选标签后点云为空。")
    # 2) 点云法向（可选）
    pc_normals = None
    if normal_consistency_deg is not None:
        pcd_n = o3d.geometry.PointCloud()
        pcd_n.points = o3d.utility.Vector3dVector(use_xyz)
        pcd_n.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )
        pc_normals = np.asarray(pcd_n.normals)  # (M,3), 单位向量

    # 3) mesh 顶点法向（用于法向一致性）
    V = mesh.vertices
    mesh_nrm = None
    if normal_consistency_deg is not None:
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(V):
            mesh.rezero()  # 可选
            mesh.compute_vertex_normals()
        mesh_nrm = np.asarray(mesh.vertex_normals, dtype=np.float64)
        # 归一化
        nn = np.linalg.norm(mesh_nrm, axis=1, keepdims=True)+1e-12
        mesh_nrm = mesh_nrm / nn

    # 4) KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(use_xyz.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    # 5) 主循环
    out = np.zeros((V.shape[0], 3), np.uint8)
    dists = np.full(V.shape[0], np.nan, float)
    cos_thresh = None
    if normal_consistency_deg is not None:
        cos_thresh = np.cos(np.deg2rad(normal_consistency_deg))

    for i, p in enumerate(V):
        # 邻域检索
        idx = None; d = None
        if neighbor_mode == 'radius' and radius is not None:
            ok, ids, dist2 = kdt.search_radius_vector_3d(p.astype(np.float64), radius)
            if ok == 0:
                out[i] = (128,128,128); continue
            idx = np.asarray(ids, dtype=int)
            d = np.sqrt(np.asarray(dist2, dtype=float))
        else:
            k = min(knn, len(use_xyz))
            ok, ids, dist2 = kdt.search_knn_vector_3d(p.astype(np.float64), k)
            if ok == 0:
                out[i] = (128,128,128); continue
            idx = np.asarray(ids, dtype=int)
            d = np.sqrt(np.asarray(dist2, dtype=float))

        # 距离硬阈值
        if max_dist is not None:
            m = d <= max_dist
            idx, d = idx[m], d[m]
            if idx.size == 0:
                out[i] = (128,128,128); continue

        # 修剪（丢最远的 trim_frac）
        if trim_frac > 0.0 and idx.size > 1:
            keep = int(np.ceil(idx.size * (1.0 - trim_frac)))
            ords = np.argsort(d)
            idx, d = idx[ords[:keep]], d[ords[:keep]]

        # 法向一致性（可选）
        if cos_thresh is not None and pc_normals is not None and mesh_nrm is not None:
            mn = mesh_nrm[i]
            cosang = (pc_normals[idx] @ mn).clip(-1,1)  # 近似比较
            m = np.abs(cosang) >= cos_thresh
            idx, d = idx[m], d[m]
            if idx.size == 0:
                out[i] = (128,128,128); continue

        # 语义一致性（only_labels 过滤已经在源头做了；如果只传 labels 而没传 only_labels，也可按多数）
        if labels is not None and only_labels is not None and use_lab is not None:
            inlier = np.isin(use_lab[idx], only_labels)
            if inlier.size == 0 or (inlier.sum() / inlier.size) < min_inlier_frac:
                out[i] = (128,128,128); continue
            # 只用 inlier 集合参与加权
            idx, d = idx[inlier], d[inlier]

        # 距离加权
        w = 1.0 / (d + 1e-8)
        w = w / w.sum()
        col = (use_rgb[idx].astype(np.float32) * w[:,None]).sum(0)
        out[i] = col.clip(0,255).astype(np.uint8)
        dists[i] = float(d.min()) if d.size else np.nan

    metrics = {
        "coverage_ratio": float(np.isfinite(dists).mean()),
        "mean_nn_distance": float(np.nanmean(dists)),
        "median_nn_distance": float(np.nanmedian(dists)),
        "over_threshold_ratio": float(np.mean(dists > (max_dist if max_dist else np.inf)))
    }
    return out, metrics

def paint_label_vertex(mesh, pc_xyz, pc_labels, palette_rgb,
                       neighbor_mode='radius', radius=0.006, knn=8,
                       max_dist=0.010, trim_frac=0.5,
                       only_labels=None, min_inlier_frac=0.7,
                       normal_consistency_deg=30.0):
    """
    给每个 mesh 顶点分配一个离散标签（多数投票），再按调色板上纯色。
    返回：vertex_colors(uint8 Nx3), vertex_labels(int Nx)
    """
    import numpy as np
    assert o3d is not None, "need open3d"
    use_idx = np.arange(len(pc_xyz))
    if only_labels is not None and pc_labels is not None:
        m = np.isin(pc_labels, only_labels)
        use_idx = use_idx[m]
    if use_idx.size == 0:
        raise ValueError("筛选后点云为空。")
    P = pc_xyz[use_idx]
    L = pc_labels[use_idx] if pc_labels is not None else None

    # KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    # 可选法向一致性（防跨面泄漏）
    pc_normals = None
    mesh_nrm = None
    if normal_consistency_deg is not None:
        pcd_n = o3d.geometry.PointCloud()
        pcd_n.points = o3d.utility.Vector3dVector(P)
        pcd_n.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pc_normals = np.asarray(pcd_n.normals)
        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.rezero()
            mesh.compute_vertex_normals()
        mesh_nrm = np.asarray(mesh.vertex_normals); mesh_nrm /= (np.linalg.norm(mesh_nrm,axis=1,keepdims=True)+1e-12)
        cos_thr = np.cos(np.deg2rad(normal_consistency_deg))

    V = mesh.vertices
    v_lab = -np.ones(len(V), dtype=int)
    v_col = np.zeros((len(V),3), np.uint8)

    for i,p in enumerate(V):
        if neighbor_mode == 'radius' and radius is not None:
            ok, ids, d2 = kdt.search_radius_vector_3d(p.astype(np.float64), radius)
        else:
            ok, ids, d2 = kdt.search_knn_vector_3d(p.astype(np.float64), min(knn, len(P)))
        if ok == 0:
            continue
        ids = np.asarray(ids, int)
        d   = np.sqrt(np.asarray(d2, float))

        # 距离硬阈值
        if max_dist is not None:
            m = d <= max_dist
            ids, d = ids[m], d[m]
            if ids.size == 0:
                continue

        # 修剪最远
        if trim_frac > 0 and ids.size > 1:
            keep = int(np.ceil(ids.size * (1.0 - trim_frac)))
            ords = np.argsort(d)
            ids, d = ids[ords[:keep]], d[ords[:keep]]

        # 法向一致性
        if pc_normals is not None:
            cosang = np.abs(pc_normals[ids] @ mesh_nrm[i])
            m = cosang >= cos_thr
            ids, d = ids[m], d[m]
            if ids.size == 0:
                continue

        # 多数投票（按 1/d 加权，仍产出“离散标签”）
        if L is None:
            # 无标签则回退到最近邻的颜色
            v_col[i] = palette_rgb.get(0, np.array([200,200,200],np.uint8))
            v_lab[i] = 0
            continue
        cand = L[ids]
        # 语义一致性：邻域中属于允许集合的比例
        if only_labels is not None:
            frac = np.isin(cand, only_labels).sum() / cand.size
            if frac < min_inlier_frac:
                continue
        # 得分 = sum_w(label==k)，其中 w=1/(d+eps)
        w = 1.0/(d+1e-8); w/=w.sum()
        uniq, inv = np.unique(cand, return_inverse=True)
        scores = np.bincount(inv, weights=w)
        k = uniq[np.argmax(scores)]
        v_lab[i] = int(k)
        v_col[i] = palette_rgb[int(k)]

    # 未赋色的顶点，用最近已赋色顶点的标签填补（一次邻接传播）
    if (v_lab<0).any():
        import scipy.sparse as sp
       # 用 CSR 邻接矩阵补洞（两轮多数投票）
        try:
            A = mesh.vertex_adjacency_matrix.tocsr()   # trimesh 提供
        except Exception:
            import networkx as nx
            G = mesh.vertex_adjacency_graph
            A = nx.to_scipy_sparse_array(G, format='csr')

        known = (v_lab >= 0)
        for _ in range(2):
            pick = (~known) & (A.dot(known.astype(np.float32)) > 0)
            if not np.any(pick): break
            for vi in np.where(pick)[0]:
                start, end = A.indptr[vi], A.indptr[vi+1]
                nbr = A.indices[start:end]
                nbr = nbr[known[nbr]]
                if nbr.size:
                    lab = np.bincount(v_lab[nbr]).argmax()
                    v_lab[vi] = lab
                    v_col[vi] = palette_rgb[int(lab)]
            known = (v_lab >= 0)


    return v_col, v_lab

def palette_from_pointcloud(all_labels, pc_labels, pc_rgb, mode='mean'):
    pal = {}
    labs = sorted(int(x) for x in set(all_labels))
    for k in labs:
        m = (pc_labels == k)
        if not np.any(m):
            pal[k] = np.array([180,180,180], np.uint8); continue
        cols = pc_rgb[m].astype(np.float32)
        if mode == 'mean':
            c = cols.mean(0)
        else:  # medoid
            mean = cols.mean(0, keepdims=True)
            idx = np.argmin(np.linalg.norm(cols - mean, axis=1))
            c = cols[idx]
        pal[k] = c.clip(0,255).astype(np.uint8)
    return pal

def paint_label_facecut(mesh, pc_xyz, pc_labels, palette_rgb,
                        neighbor_mode='radius', radius=0.006, knn=8,
                        max_dist=0.010, trim_frac=0.5,
                        only_labels=None, min_inlier_frac=0.7):
    import numpy as np
    assert o3d is not None
    
    use_idx = np.arange(len(pc_xyz))
    if only_labels is not None and pc_labels is not None:
        m = np.isin(pc_labels, only_labels)
        use_idx = use_idx[m]
    if use_idx.size == 0:
        raise ValueError("筛选后点云为空。")
    P = pc_xyz[use_idx]; L = pc_labels[use_idx] if pc_labels is not None else None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    V = mesh.vertices; F = mesh.faces
    face_lab = np.full(len(F), -1, int)

    # 1) 面心判标签（多数投票）
    C = V[F].mean(axis=1)  # (M,3)
    for i,c in enumerate(C):
        if neighbor_mode=='radius' and radius is not None:
            ok, ids, d2 = kdt.search_radius_vector_3d(c.astype(np.float64), radius)
        else:
            ok, ids, d2 = kdt.search_knn_vector_3d(c.astype(np.float64), min(knn, len(P)))
        if ok==0: continue
        ids = np.asarray(ids,int); d = np.sqrt(np.asarray(d2,float))
        if max_dist is not None:
            m = d <= max_dist; ids,d = ids[m], d[m]
            if ids.size==0: continue
        if trim_frac>0 and ids.size>1:
            keep = int(np.ceil(ids.size*(1.0-trim_frac)))
            ords = np.argsort(d); ids,d = ids[ords[:keep]], d[ords[:keep]]
        if L is None: 
            face_lab[i] = 0; continue
        cand = L[ids]
        if only_labels is not None:
            frac = np.isin(cand, only_labels).sum()/cand.size
            if frac < min_inlier_frac: continue
        uniq, inv = np.unique(cand, return_inverse=True)
        w = 1.0/(d+1e-8); w/=w.sum()
        scores = np.bincount(inv, weights=w)
        face_lab[i] = int(uniq[np.argmax(scores)])

    # 2) 重建“按面复制顶点”的网格，并按面上纯色
    new_vertices = V[F].reshape(-1,3)            # 每面3个新顶点
    new_faces    = np.arange(len(new_vertices)).reshape(-1,3)
    new_colors   = np.zeros((len(new_vertices),3), np.uint8)
    for i,lab in enumerate(face_lab):
        col = palette_rgb.get(int(lab), np.array([180,180,180],np.uint8))
        new_colors[i*3:(i+1)*3] = col
    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    
    return new_mesh, new_colors, face_lab


def export_glb_with_color(mesh, vertex_rgb, out_path):
    from trimesh.visual import ColorVisuals
    from trimesh.exchange import gltf
    if hasattr(mesh.visual,'uv'):       mesh.visual.uv = None
    if hasattr(mesh.visual,'material'): mesh.visual.material = None
    vc = np.zeros((len(mesh.vertices),4), np.uint8)
    vc[:,:3] = vertex_rgb; vc[:,3]=255
    mesh.visual = ColorVisuals(mesh, vertex_colors=vc)
    glb_bytes = gltf.export_glb(mesh, include_normals=True)
    with open(out_path, 'wb') as f: f.write(glb_bytes)
def export_obj_with_vertex_colors(mesh, vertex_rgb, out_obj_path):
    """
    直接把每个顶点的 RGB 写进 OBJ 的 v 行（v x y z r g b，r/g/b ∈ [0,1]）
    faces 用 f v//vn 形式（不写 vt）
    绝大多数现代 DCC（Blender 4.x、MeshLab 等）能正确显示顶点色。
    """
    import numpy as np, os
    os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=int)
    # 法向（有就用，没有就算）
    if getattr(mesh, "vertex_normals", None) is None or len(mesh.vertex_normals) != len(V):
        mesh.compute_vertex_normals()
    N = np.asarray(mesh.vertex_normals, dtype=float)

    C = np.asarray(vertex_rgb, dtype=float)
    if C.max() > 1.0: C = C / 255.0  # 归一化到[0,1]

    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.write("# OBJ with per-vertex color extension\n")
        # 顶点 + 颜色
        for i in range(len(V)):
            x,y,z = V[i]
            r,g,b  = C[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        # 法向
        for i in range(len(N)):
            nx,ny,nz = N[i]
            f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
        # 面（不写 vt，直接 v//vn）
        for (a,b,c) in F:
            ia, ib, ic = a+1, b+1, c+1
            f.write(f"f {ia}//{ia} {ib}//{ib} {ic}//{ic}\n")

def patch_mtl_unlit(mtl_path, tex_rel='textures/albedo.png', force_emissive_alias=True):
    """
    将 Blender 导出的 .mtl 修改为“近似无光照”：
    - Kd/Ka 设为 1 1 1，Ks=0，illum=0（或1）
    - map_Kd 指向贴图；（可选）补一条 map_Ka 同贴图，增稳
    - 可选：增加非标准但常见的 Ke/map_Ke（有些查看器把它当发光）
    """
    if not os.path.isfile(mtl_path): return
    with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    out = []
    seen_map_kd = False
    seen_map_ka = False
    for line in lines:
        s = line.strip()
        if s.startswith('Kd '):
            line = 'Kd 1.000 1.000 1.000\n'
        elif s.startswith('Ka '):
            line = 'Ka 1.000 1.000 1.000\n'
        elif s.startswith('Ks '):
            line = 'Ks 0.000 0.000 0.000\n'
        elif s.startswith('Ns '):
            line = 'Ns 0.000\n'
        elif s.startswith('illum '):
            line = 'illum 0\n'   # 不做高光/折射，尽量无光
        elif s.startswith('map_Kd'):
            line = f'map_Kd {tex_rel}\n'
            seen_map_kd = True
        elif s.startswith('map_Ka'):
            line = f'map_Ka {tex_rel}\n'
            seen_map_ka = True
        out.append(line)

    # 若没 map_Ka，补一条（很多查看器把 Ka 作为“环境色”叠加）
    if seen_map_kd and not seen_map_ka:
        # 插到第一条 map_Kd 后
        for i,l in enumerate(out):
            if l.startswith('map_Kd'):
                out.insert(i+1, f'map_Ka {tex_rel}\n')
                break

    # （可选）补发光，部分查看器支持 Ke/map_Ke，当作“自发光”不受光照
    if force_emissive_alias:
        has_ke = any(l.startswith('Ke ') for l in out)
        has_mapke = any(l.startswith('map_Ke') for l in out)
        if not has_ke:
            out.append('Ke 1.000 1.000 1.000\n')
        if not has_mapke and seen_map_kd:
            out.append(f'map_Ke {tex_rel}\n')

    with open(mtl_path, 'w', encoding='utf-8') as f:
        f.writelines(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pc', required=True, help='points.ply')
    ap.add_argument('--glb', required=True, help='input model.glb')
    ap.add_argument('--workdir', default=None,
                help='可选：中间文件目录；缺省时使用临时目录并自动清理')
    
    ap.add_argument('--rx_deg', type=float, default=0.0, help='Blender 恢复性旋转（绕X，度）')
    ap.add_argument('--pivot', choices=['pc_centroid','mesh_centroid','origin'], default='pc_centroid')
    ap.add_argument('--blender', default='blender')
    ap.add_argument('--knn', type=int, default=8)
    ap.add_argument('--max_dist', type=float, default=0.03)
    ap.add_argument('--icp', action='store_true', help='颜色回传前做点到点ICP微调')
    ap.add_argument('--make_obj', action='store_true', help='烘焙贴图并导出 OBJ+MTL+textures')
    ap.add_argument('--tex_res', type=int, default=2048)
    ap.add_argument('--labels', default=None, help='npy: 每点的语义/部位标签')
    ap.add_argument('--only_labels', type=int, nargs='*', default=None,
                    help='仅从这些标签的点上色（例如把手=1）')
    ap.add_argument('--neighbor_mode', choices=['knn','radius'], default='knn',
                    help='邻域检索方式：kNN或半径搜索')
    ap.add_argument('--radius', type=float, default=None,
                    help='radius 模式的半径（米），如 0.008；若设定则优先于 knn')
   
    ap.add_argument('--out_glb',  type=str, default=None,
                    help='最终着色后的 GLB 完整路径（含文件名）。若缺省，落到 <workdir>/colored_aligned.glb')
    ap.add_argument('--report',   type=str, default=None,
                    help='单一 JSON 报告完整路径（含文件名），会合并 align 与 color-transfer 两部分')
    ap.add_argument('--obj_mode', choices=['none','vcol','baked'], default='none',
                    help='OBJ 导出方式：vcol=顶点色OBJ；baked=烘焙贴图OBJ；none=不导出OBJ')
    ap.add_argument('--obj_out',  type=str, default=None,
                    help='当 obj_mode=vcol 时，OBJ 完整输出路径（含文件名）。若缺省，落到 <workdir>/_obj/model_vcol.obj')
    ap.add_argument('--obj_dir',  type=str, default=None,
                    help='当 obj_mode=baked 时，OBJ 输出目录（包含 model.obj、textures/、MTL）。若缺省，落到 <workdir>/_obj')
    ap.add_argument('--keep_intermediate', action='store_true',
                    help='保留 workdir 中间文件；缺省不保留')

    ap.add_argument('--trim_frac', type=float, default=0.3,
                    help='修剪比例：先取邻域，再丢弃最远的这部分再加权（0~0.9）')
    ap.add_argument('--min_inlier_frac', type=float, default=0.6,
                    help='邻域中属于允许 labels 的比例低于该阈值则不着色')
    ap.add_argument('--normal_consistency_deg', type=float, default=None,
                    help='若设定：要求 |angle(n_mesh, n_pc)| ≤ 阈值（度），否则剔除该邻居')
    # 上色策略：连续/离散（顶点）/离散（按面并切边界）
    ap.add_argument('--paint_mode', choices=['rgb_weighted','label_vertex','label_facecut'],
                default='label_vertex',
                help='rgb_weighted=距离加权RGB；label_vertex=顶点多数标签+调色板；label_facecut=按面判标签并在标签边界切开')
    ap.add_argument('--base_rgb', type=str, default="200,200,200",
                help='烘焙时的底色（R,G,B），如 180,180,180')
    ap.add_argument('--vcol_threshold', type=float, default=0.02,
                help='烘焙时的顶点色阈值（0~1），低于该阈值的像素用底色填充')
    ap.add_argument('--obj_mode', choices=['baked','vcol'], default='baked',
                help='OBJ 导出方式：baked=烘焙贴图；vcol=顶点色写入 OBJ 顶点行（最像 GLB）')

    args = ap.parse_args()

    import tempfile
    from pathlib import Path
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        _tmp = None
    else:
        _tmp = tempfile.TemporaryDirectory()
        workdir = Path(_tmp.name)

    with tempfile.TemporaryDirectory() as td:
        # 写入两个 blender 子脚本
        align_py = write_tmp_script(BLENDER_ALIGN_PY, 'blender_align.py', td)
        bake_py  = write_tmp_script(BLENDER_BAKE_PY,  'bake_vcol_to_obj.py', td)

        # Step1: Blender 恢复性旋转 + 报告
        aligned_glb = str(workdir / 'aligned.glb')  # 中间产物仍在 workdir
        align_report = str(workdir / 'align_report.json')  # 先各写到 workdir，末尾再合并到 --report
        # 读取点云，算质心（在主脚本 Python 内）
        pc_xyz, pc_rgb = load_point_cloud(args.pc)
        pc_centroid = pc_xyz.mean(axis=0).tolist()

        # 调 Blender 对齐（注意：不再传 --in_ply）
        sh([args.blender, "-b", "-P", align_py, "--",
        "--in_glb", os.path.abspath(args.glb),
        "--rx_deg", str(args.rx_deg),
        "--pivot", args.pivot,  # 仍保留枚举，用于 mesh_centroid/origin 情况
        "--pivot_xyz", f"{pc_centroid[0]},{pc_centroid[1]},{pc_centroid[2]}",
        "--out_glb", os.path.abspath(aligned_glb),
        "--report", os.path.abspath(align_report)
        ])

        # Step2: 点云→网格 顶点色
        pc_xyz, pc_rgb = load_point_cloud(args.pc)
        mesh = load_mesh(aligned_glb)
        labels = np.load(args.labels) if args.labels else None
        # 可选 ICP 微调（在 aligned.glb 基础上进一步细调）
        if args.icp:
            if o3d is None: raise RuntimeError("open3d required for --icp")
            T = icp_refine(mesh, pc_xyz, voxel=0.005, max_iter=80)
            mesh.apply_transform(T)

        labels = np.load(args.labels) if args.labels else None

        # 调色板（基于 labels.npy 出现过的标签生成）
        used = np.unique(labels) if labels is not None else [0]
        palette = palette_from_pointcloud(used, labels, pc_rgb, mode='mean')        

        if args.paint_mode == 'rgb_weighted':
            # 旧方式（连续RGB）；不建议在边界清晰的目标上使用
            vcols, metrics = color_transfer(  # 你原来的连续上色函数
                pc_xyz, pc_rgb, mesh,
                neighbor_mode=args.neighbor_mode,
                knn=args.knn, radius=args.radius, max_dist=args.max_dist,
                labels=labels, only_labels=args.only_labels,
                trim_frac=args.trim_frac, min_inlier_frac=args.min_inlier_frac,
                normal_consistency_deg=args.normal_consistency_deg
            )
            mesh_out = mesh.copy()
            colored_glb = os.path.join(args.workdir, "colored_aligned.glb")

        elif args.paint_mode == 'label_vertex':
            vcols, vlab = paint_label_vertex(
                mesh, pc_xyz, labels, palette,
                neighbor_mode=args.neighbor_mode, radius=args.radius, knn=args.knn,
                max_dist=args.max_dist, trim_frac=args.trim_frac,
                only_labels=args.only_labels, min_inlier_frac=args.min_inlier_frac,
                normal_consistency_deg=args.normal_consistency_deg
            )
            mesh_out = mesh.copy()
            mask = (vlab >= 0)
            metrics = compute_metrics_for_vertices(pc_xyz, mesh_out.vertices, mask, args.max_dist)
            colored_glb = os.path.join(args.workdir, "colored_aligned.glb")

        elif args.paint_mode == 'label_facecut':
            mesh_out, vcols, flab = paint_label_facecut(
                mesh, pc_xyz, labels, palette,
                neighbor_mode=args.neighbor_mode, radius=args.radius, knn=args.knn,
                max_dist=args.max_dist, trim_frac=args.trim_frac,
                only_labels=args.only_labels, min_inlier_frac=args.min_inlier_frac
            )
            mask = np.ones(len(mesh_out.vertices), dtype=bool)   # facecut 每面都给了颜色
            metrics = compute_metrics_for_vertices(pc_xyz, mesh_out.vertices, mask, args.max_dist)
            colored_glb = os.path.join(args.workdir, "colored_aligned.glb")
        else:
            raise ValueError("unknown paint_mode")

        # 导出 GLB（COLOR_0 顶点色），后续照常烘焙->OBJ
        BASE = np.array([200,200,200], np.uint8)   # 底色可做成参数

        if args.only_labels is not None and 'vlab' in locals():
            keep = np.isin(vlab, np.array(args.only_labels))
            vcols[~keep] = BASE 

        # 导出带 COLOR_0 的 GLB（供查看/烘焙）
        ccolored_glb = args.out_glb if args.out_glb else str(workdir / 'colored_aligned.glb')
        export_glb_with_color(mesh_out, vcols, colored_glb)

        ct_report = os.path.join(args.workdir, "ct_report.json")
        with open(ct_report, "w") as f: json.dump(metrics, f, indent=2)
        print("[report]", json.dumps(metrics, indent=2))
        obj_dir = os.path.join(args.workdir, "_obj")
        os.makedirs(obj_dir, exist_ok=True)
        # Step3: 烘焙 & 导出 OBJ+MTL+textures
        if args.make_obj and args.obj_mode == 'baked':
            sh([args.blender, "-b", "-P", bake_py, "--",
                "--in_mesh", os.path.abspath(colored_glb),
                "--out_dir", os.path.abspath(args.workdir),
                "--tex_name", "albedo",
                "--tex_res", str(args.tex_res),
                "--attr_name", "COLOR_0",
                "--base_rgb", args.base_rgb,              # 让底色一致
                "--vcol_threshold", str(args.vcol_threshold),
                "--uv_margin", "0.03",
                "--per_face_islands", "true"              # 硬边优先
            ])
        else:
            # vcol：直接写 OBJ（顶点色）
            out_obj = os.path.join(obj_dir, "model_vcol.obj")
            export_obj_with_vertex_colors(mesh_out, vcols, out_obj)
            print("[OK] OBJ (vertex-colored) exported:", out_obj)
        mtl_path = os.path.join(obj_dir, "model.mtl")
        patch_mtl_unlit(mtl_path, tex_rel='textures/albedo.png', force_emissive_alias=True)

        print("\n[OK] All done.")
        print("Aligned GLB: ", aligned_glb)
        print("Colored GLB:", colored_glb)
        if args.make_obj:
            print("OBJ dir    :", os.path.join(args.workdir, "obj"))
        print("Reports    :", align_report, "|", ct_report)

if __name__ == "__main__":
    main()
