# align_export_snapshot_blender.py
# 批量：按 JSON 的 3x3 旋转矩阵对齐每个 shape，(可选) 归一到统一尺度，导出 GLB。
# 然后把所有 GLB 摆在 NxN 中心对称网格上，自动布相机并渲染一张 snapshot.png。
import bpy, sys, os, json, argparse, math
import bmesh
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector
from pathlib import Path
import subprocess, tempfile

PRIORITY_EXTS = [".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"]
REQUIRE_HOLLOW_DEFAULT = {"mug", "kettle", "teapot", "pitcher"}

import contextlib
from math import inf

from collections.abc import Iterable

def _as_iter(objs):
    """把输入规范化为 list；若是单个 Object 就包成 [obj]。"""
    if objs is None:
        return []
    # 已经是可迭代（例如 list/tuple/bpy_prop_collection）
    try:
        if isinstance(objs, Iterable) and not isinstance(objs, (str, bytes)):
            return list(objs)
    except Exception:
        pass
    # 单个对象
    return [objs]

def _gather_renderable_meshes(objs):
    """展开父子层级，拿到所有可渲染的 MESH 对象（忽略 EMPTY/LIGHT/CAMERA 等）。"""
    out, seen, stack = [], set(), _as_iter(objs)
    while stack:
        o = stack.pop()
        if not o or o.name in seen:
            continue
        seen.add(o.name)
        if getattr(o, "type", None) == 'MESH':
            out.append(o)
        # 展开子层级
        for ch in getattr(o, "children", []):
            stack.append(ch)
    return out

# def _world_bbox_max_dim(mesh_objs, depsgraph):
#     """用评估后的世界几何计算 max_dim（最稳妥，含修改器/实例）。返回 (max_dim, per_obj_dims[list])"""
#     import mathutils
#     dims = []
#     minv = mathutils.Vector(( inf,  inf,  inf))
#     maxv = mathutils.Vector((-inf, -inf, -inf))
#     for ob in mesh_objs:
#         ob_eval = ob.evaluated_get(depsgraph)
#         # to_mesh 包含修改器等
#         me = ob_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
#         if not me or len(me.vertices) == 0:
#             if me: ob_eval.to_mesh_clear()
#             continue
#         M = ob_eval.matrix_world
#         # 同时记录每个对象自身的包围盒尺寸，后面做离群过滤用
#         lo = mathutils.Vector(( inf,  inf,  inf))
#         hi = mathutils.Vector((-inf, -inf, -inf))
#         for v in me.vertices:
#             w = M @ v.co
#             # 总体
#             minv.x = min(minv.x, w.x);  minv.y = min(minv.y, w.y);  minv.z = min(minv.z, w.z)
#             maxv.x = max(maxv.x, w.x);  maxv.y = max(maxv.y, w.y);  maxv.z = max(maxv.z, w.z)
#             # 单对象
#             lo.x = min(lo.x, w.x); lo.y = min(lo.y, w.y); lo.z = min(lo.z, w.z)
#             hi.x = max(hi.x, w.x); hi.y = max(hi.y, w.y); hi.z = max(hi.z, w.z)
#         dims.append((ob, (hi - lo)))
#         ob_eval.to_mesh_clear()
#     overall = (maxv - minv)
#     max_dim = max(overall.x, overall.y, overall.z) if dims else 0.0
#     return max_dim, dims

# def _world_bbox_minmax(mesh_objs, depsgraph):
#     import mathutils
#     minv = mathutils.Vector(( inf,  inf,  inf))
#     maxv = mathutils.Vector((-inf, -inf, -inf))
#     for ob in mesh_objs:
#         ob_eval = ob.evaluated_get(depsgraph)
#         me = ob_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)
#         if not me or len(me.vertices) == 0:
#             if me: ob_eval.to_mesh_clear()
#             continue
#         M = ob_eval.matrix_world
#         for v in me.vertices:
#             w = M @ v.co
#             if w.x < minv.x: minv.x = w.x
#             if w.y < minv.y: minv.y = w.y
#             if w.z < minv.z: minv.z = w.z
#             if w.x > maxv.x: maxv.x = w.x
#             if w.y > maxv.y: maxv.y = w.y
#             if w.z > maxv.z: maxv.z = w.z
#         ob_eval.to_mesh_clear()
#     return minv, maxv

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = 'METRIC'
    scn.unit_settings.scale_length = 1.0

# def find_mesh_file(inst_dir: Path) -> str | None:
#     RAW_DIR = inst_dir/"raw"
#     for ext in PRIORITY_EXTS:
#         cands = sorted(RAW_DIR.rglob(f"*{ext}"), key=lambda p: (len(p.parts), len(str(p))))
#         if cands:
#             return str(cands[0])
#     return None
def find_mesh_file(inst_dir: Path) -> str | None:
    RAW_DIR = inst_dir/"raw"
    # 先优先 fixed 的 OBJ
    fixed = sorted(RAW_DIR.rglob("*_fixed.obj"), key=lambda p: (len(p.parts), len(str(p))))
    if fixed:
        return str(fixed[0])
    # 其次按优先扩展找其它格式（含普通 .obj）
    for ext in PRIORITY_EXTS:
        cands = sorted(RAW_DIR.rglob(f"*{ext}"), key=lambda p: (len(p.parts), len(str(p))))
        if cands:
            return str(cands[0])
    return None

def _is_ascii_fbx(path: Path) -> bool:
    with open(path, 'rb') as f:
        head = f.read(32)
    # 二进制 FBX 文件头包含 'Kaydara FBX Binary'
    return b'Kaydara FBX Binary' not in head

def _convert_ascii_fbx_to_glb(fbx_path: Path, fbx2glb_bin="fbx2glb"):
    """Convert FBX (ASCII or binary) to a temporary GLB using fbx2glb."""
    out_glb = Path(tempfile.gettempdir()) / (fbx_path.stem + ".glb")
    try:
        subprocess.run(
            [fbx2glb_bin, str(fbx_path), str(out_glb), "--khr-materials-unlit"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if out_glb.exists():
            return out_glb
    except Exception as e:
        print(f"[FBX2GLB FAIL] {fbx_path} -> {e}")
    return None

from pathlib import Path
import re

def _is_real_file_image(img) -> bool:
    """
    仅把“真文件”图片视为可重连/可打包对象：
      - 有有效的 filepath（带扩展名）
      - 非 'Map #1' / 'Tex #2' 这类占位名（可带或不带扩展名）
      - 源必须是 FILE
    """
    if not img:
        return False
    p = (img.filepath or "").strip()
    if not p:
        return False
    bn = Path(p).name
    # 占位名：Map #1 / Tex #2（可带扩展名）
    if re.match(r'^(map|tex)\s*#\d+(?:\.\w+)?$', bn, flags=re.I):
        return False
    # 必须有扩展名
    if not Path(p).suffix:
        return False
    try:
        if hasattr(img, "source") and img.source != 'FILE':
            return False
    except Exception:
        pass
    return True
# 1) 统一把图片从 Windows 绝对路径/杂路径，重定向到本地 raw / <fbx>.fbm
import os, re, bpy
from pathlib import Path

WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")
def _looks_foreign(p): return bool(WIN_ABS_RE.match(p or ""))

def relink_images(raw_dir: Path, fbm_dir: Path|None=None):
    fixed = missing = 0
    for img in bpy.data.images:
        if not img or not img.filepath: continue
        fp_abs = bpy.path.abspath(img.filepath, library=img.library)
        # 已经是有效本地文件就跳过
        if os.path.isfile(fp_abs) and not _looks_foreign(img.filepath): 
            continue
        # 以 basename 在常见位置查找
        bn = os.path.basename(fp_abs or img.name)
        cands = []
        if fbm_dir: cands.append(fbm_dir / bn)
        cands += [Path(raw_dir) / "textures" / bn, Path(raw_dir) / bn]
        hit = next((c for c in cands if c.exists()), None)
        if hit:
            img.filepath = str(hit)
            try: img.reload()
            except: pass
            fixed += 1
        else:
            missing += 1
    print(f"[align] relink: fixed={fixed}, missing={missing}")

def relink_images_from_fbm(fbx_path: Path):
    """
    FBX 导入后，尝试把找不到的贴图重连到：
      1) <FBX>.fbm/<basename>
      2) <FBX同目录>/<basename>
    先跑一次 Blender 的 Find Missing Files（等价 GUI: External Data -> Find Missing Files）。
    """
    import bpy, os
    fbx_path = Path(fbx_path)
    fbx_dir  = fbx_path.parent
    fbm_dir  = fbx_path.with_suffix(".fbm")

    # 先让 Blender 自己找一遍（基于 FBX 所在目录）
    try:
        bpy.ops.file.find_missing_files(directory=str(fbx_dir))
    except Exception:
        pass

    # 再按 basename 到 .fbm / 同目录兜底
    for img in list(bpy.data.images):
        if not _is_real_file_image(img):
            continue  # ← 新增：占位图/无扩展/非FILE 直接跳过
        if not img:
            continue
        # 尝试解析当前路径（可能为空/无效/绝对路径）
        cand_name = None
        try:
            raw = img.filepath_raw or img.filepath
            if raw:
                cand_name = Path(bpy.path.abspath(raw)).name
        except Exception:
            pass
        if not cand_name:
            # 没有路径就用节点名兜底（常见：打包图像或丢失）
            cand_name = (img.name if "." in img.name else img.name + ".png")

        # 逐目录尝试：.fbm -> FBX 同目录
        for base in (fbm_dir, fbx_dir):
            p = base / cand_name
            if p.exists():
                try:
                    img.filepath = bpy.path.relpath(str(p))  # 写相对路径更可移植
                    img.reload()
                except Exception:
                    pass
                break

    # 统一把图片路径相对化 —— 先确保已保存
    try:
        if not bpy.data.is_saved:
            tmp_blend = fbx_path.with_suffix(".temp.blend")
            bpy.ops.wm.save_mainfile(filepath=str(tmp_blend))
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_mainfile()  # 保存当前相对化后的状态
        bpy.ops.file.pack_all()
    except Exception:
        pass

def sanitize_and_relink_images(base_dir: str,
                               fbx_path: str | None = None,
                               extra_roots: list[str] | None = None,
                               prefer_textures_sub=True):
    """
    统一修复贴图路径：
    - 若路径为绝对（含 Windows 盘符或以/开头），仅取 basename；
    - 在 [<fbx>.fbm, <raw>, <raw>/textures, extra_roots...] 里按 basename 查找；
    - 命中后设置为相对路径（相对 base_dir）；
    - 未命中则不打包，留warning。
    """
    import bpy, os, re
    from pathlib import Path

    WIN_ABS = re.compile(r"^[A-Za-z]:[\\/]|^\\\\")
    base_dir = Path(base_dir).resolve()
    roots: list[Path] = []

    if fbx_path:
        fbxp = Path(fbx_path)
        roots += [fbxp.with_suffix(".fbm"), fbxp.parent]

    # 推断一个 raw 根（常见你的目录结构里：实例 raw）
    # base_dir / "../../datasets/Objaverse.../<inst>/raw"
    # 也可以在上层调用处显式传入
    for r in (extra_roots or []):
        roots.append(Path(r))
        if prefer_textures_sub:
            roots.append(Path(r) / "textures")

    def is_abs_win_or_unix(p: str) -> bool:
        return p.startswith("/") or bool(WIN_ABS.match(p))

    for img in list(bpy.data.images):
        try:
            # 只处理真正落盘的图像
            if img.packed_file or not (img.source in {'FILE', 'SEQUENCE'}):
                continue

            raw = (img.filepath_raw or img.filepath or "").strip()
            if not raw:
                continue
            # Blender 可能返回 // 相对写法
            abspath = Path(bpy.path.abspath(raw)).as_posix()

            # 取 basename（若是绝对路径或 Windows 路径）
            bn = Path(abspath).name if is_abs_win_or_unix(abspath) else Path(raw).name
            # 某些图像名字没扩展名，给一个兜底
            if "." not in bn and "." in img.name:
                bn = img.name

            # 逐根目录尝试命中
            hit = None
            for root in roots:
                # 先直接拼接
                cand = root / bn
                if cand.exists():
                    hit = cand; break
                # 再做一次浅搜（避免深度 rglob 太慢）
                for sub in [root, root / "textures"]:
                    cand = sub / bn
                    if cand.exists():
                        hit = cand; break
                if hit: break

            if hit and hit.is_file():
                # 设为相对 base_dir 的路径
                rel = os.path.relpath(str(hit), str(base_dir))
                img.filepath = bpy.path.abspath(rel)  # 设置绝对，再转相对更稳
            else:
                print(f"[warn] image not found for basename: {bn} (from {raw})")
        except Exception as e:
            print(f"[warn] sanitize image failed: {getattr(img,'name', '')}: {e}")

    # 显式相对化到 base_dir，避免“未保存 blend”的警告
    try:
        bpy.ops.file.make_paths_relative(base_directory=str(base_dir))
    except Exception as e:
        print(f"[warn] make_paths_relative failed: {e}")

def import_mesh(path: str):
    p = Path(path); ext = p.suffix.lower()
    if ext == ".fbx":
        if _is_ascii_fbx(p):
            # 自动转换到临时目录
            # with tempfile.TemporaryDirectory() as td:
            out = _convert_ascii_fbx_to_glb(p, fbx2glb_bin="fbx2glb")
            if not out:
                raise RuntimeError(f"fbx2glb_failed: {p}")
            bpy.ops.import_scene.gltf(filepath=path, 
                    merge_vertices=True,
                    bone_heuristic='TEMPERANCE',      # 稳一点的骨骼启发
                    guess_original_bind_pose=True,    # 让导入尽量还原 Bind Pose)
                    import_pack_images=False,
            )
        else:
            bpy.ops.import_scene.fbx(filepath=str(p), global_scale=1.0,use_anim=False, # global scale
                        use_custom_props=False,
                        ignore_leaf_bones=True,
                        automatic_bone_orientation=True, 
                        use_custom_normals=True,
                        use_image_search=True,
                        )
        # ★ 新增：导入后做一次 .fbm/ 同名兜底重连
        relink_images_from_fbm(p)
        
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=path, 
                    merge_vertices=True,
                    bone_heuristic='TEMPERANCE',      # 稳一点的骨骼启发
                    guess_original_bind_pose=True,    # 让导入尽量还原 Bind Pose)
                    import_pack_images=False, 
            )
    # elif ext == ".obj":
    #     # 若同名 .mtl 存在，则生成 _fixed.mtl，并让 .obj 指向它
    #     fixed_obj = p
    #     mtl = p.with_suffix(".mtl")
    #     if mtl.exists():
    #         fixed_mtl = fix_mtl_minimal(mtl)

    #         # 生成 _fixed.obj：只改 mtllib 行，并顺手把反斜杠替换成正斜杠
    #         lines_in  = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    #         lines_out = []
    #         for l in lines_in:
    #             if l.lower().startswith("mtllib"):
    #                 # 指向修复后的 mtl
    #                 lines_out.append(f"mtllib {fixed_mtl.name}")
    #             else:
    #                 # 保险起见全行替换一下反斜杠
    #                 lines_out.append(l.replace("\\", "/"))
    #         fixed_obj = p.with_name(p.stem + "_fixed.obj")
    #         fixed_obj.write_text("\n".join(lines_out), encoding="utf-8")
    #     try:
    #         if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
    #             bpy.ops.wm.obj_import(filepath=str(fixed_obj))
    elif ext == ".obj":
        # 若同目录存在 *_fixed.obj，优先导它；否则导原始 OBJ（不在这里再生成 fixed）
        fixed_obj = p.with_name(p.stem + "_fixed.obj")
        use_obj = fixed_obj if fixed_obj.exists() else p
        try:
            if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=str(use_obj), global_scale=1.0, )
            else:
                bpy.ops.import_scene.obj(filepath=str(use_obj), use_mtl=True)
        except Exception as e:
            raise RuntimeError(f"OBJ import failed: {e}")

    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=path)
    else:
        raise RuntimeError(f"Unsupported {ext}")
    
    return [o for o in bpy.context.selected_objects if o.type == 'MESH']

# def strip_rig_and_bake_to_mesh():
#     """
#     将当前场景中的骨骼形变烘焙到网格，然后删除 Armature/动画/相机/灯光/空物体。
#     - 把 Armature 设为 REST（避免姿势/动画影响）
#     - 对所有带 Armature/骨骼变形的 Mesh：应用 Armature Modifier（或直接 Convert→MESH）
#     - 清动画/NLA
#     - 删除非 MESH 对象（相机/灯/骨骼/空）
#     """
#     import bpy

#     # 1) Armature 统一设为 REST，切到帧0
#     bpy.context.scene.frame_set(0)
#     for arm in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
#         try:
#             arm.data.pose_position = 'REST' 
#         except Exception:
#             pass

#     # 2) 对网格应用骨骼形变（优先显式应用 Armature modifier；失败就直接 Convert）
#     meshes = [o for o in bpy.data.objects if o.type == 'MESH']
#     for m in meshes:
#         try:
#             # 逐个应用 Armature 修饰器
#             for mod in list(m.modifiers):
#                 if mod.type == 'ARMATURE':
#                     # 需要目标骨骼在 REST、且处于 OBJECT 模式
#                     bpy.ops.object.select_all(action='DESELECT')
#                     m.select_set(True)
#                     bpy.context.view_layer.objects.active = m
#                     try:
#                         bpy.ops.object.modifier_apply(modifier=mod.name)
#                     except Exception:
#                         pass
#         except Exception:
#             pass
#         # 如果还存在非几何性修饰器可能影响尺寸，保守做一次 Convert
#         if any(md.type in {'ARMATURE'} for md in getattr(m, "modifiers", [])):
#             bpy.ops.object.select_all(action='DESELECT')
#             m.select_set(True)
#             bpy.context.view_layer.objects.active = m
#             try:
#                 bpy.ops.object.convert(target='MESH', keep_original=False)
#             except Exception:
#                 pass

#     # 3) 清动画/NLA
#     for o in bpy.data.objects:
#         o.animation_data_clear() if hasattr(o, "animation_data_clear") else None

#     # 4) 删除非 MESH 对象（相机、灯、骨骼、空物体）
#     kill = [o for o in bpy.data.objects if o.type in {'ARMATURE','CAMERA','LIGHT','EMPTY'}]
#     for o in kill:
#         try:
#             bpy.data.objects.remove(o, do_unlink=True)
#         except Exception:
#             pass

# def bake_skins_and_strip_non_meshes():
#     """
#     - 把所有 Armature 的姿势设为 REST，帧设 0
#     - 对所有带 Armature 的 MESH 应用 Armature 修饰器；失败则 Convert→MESH
#     - 清动画/NLA
#     - 删除 ARMATURE/CAMERA/LIGHT/EMPTY（unlink）
#     """
#     import bpy
#     bpy.context.scene.frame_set(0)
#     # REST
#     for arm in [o for o in bpy.data.objects if o.type == 'ARMATURE']:
#         with contextlib.suppress(Exception):
#             arm.data.pose_position = 'REST' #if set to rest, will lose the original orientation, POSE

#     # 应用骨骼形变
#     for m in [o for o in bpy.data.objects if o.type == 'MESH']:
#         for md in list(getattr(m, "modifiers", [])):
#             if md.type == 'ARMATURE':
#                 bpy.ops.object.select_all(action='DESELECT')
#                 m.select_set(True); bpy.context.view_layer.objects.active = m
#                 with contextlib.suppress(Exception):
#                     bpy.ops.object.modifier_apply(modifier=md.name)
#         # 兜底：若仍残留 Armature 修饰器，直接 Convert→MESH
#         if any(md.type == 'ARMATURE' for md in getattr(m, "modifiers", [])):
#             bpy.ops.object.select_all(action='DESELECT')
#             m.select_set(True); bpy.context.view_layer.objects.active = m
#             with contextlib.suppress(Exception):
#                 bpy.ops.object.convert(target='MESH', keep_original=False)

#     # 清动画
#     for o in bpy.data.objects:
#         with contextlib.suppress(Exception):
#             o.animation_data_clear()

#     # 删除非网格
#     for o in list(bpy.data.objects):
#         if o.type in {'ARMATURE','CAMERA','LIGHT','EMPTY'}:
#             with contextlib.suppress(Exception):
#                 bpy.data.objects.remove(o, do_unlink=True)


from pathlib import Path
import re

def fix_mtl_minimal(mtl_path: Path) -> Path:
    """
    仅做两件事：
      - map_*/bump 行中的反斜杠 `\` → `/`
      - bump 行中去掉 `-bm <num>` 参数
    其余内容（Kd/illum/Tf/map_Ka/map_Kf 等）保持不变。
    返回：修复后的 _fixed.mtl 路径
    """
    if not mtl_path.exists():
        return mtl_path

    out_lines = []
    # 正则：匹配任意 -bm 及后面的数字
    bm_re = re.compile(r"\s-bm\s+[0-9]*\.?[0-9]+", re.IGNORECASE)

    for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw
        # 仅在 map_*/bump 行里做路径和 -bm 处理
        head = line.strip().lower()
        if head.startswith("map_") or head.startswith("bump"):
            # 统一路径分隔符
            line = line.replace("\\", "/")
            # 去掉 -bm 参数（仅 bump 行有效，map_* 行里一般没有 -bm）
            if head.startswith("bump"):
                line = bm_re.sub("", line)
        out_lines.append(line)

    fixed = mtl_path.with_name(mtl_path.stem + "_fixed.mtl")
    fixed.write_text("\n".join(out_lines), encoding="utf-8")
    return fixed

def fix_mtl_for_blender(mtl_path: Path):
    """
    修复 MTL：
      - `\` → `/`
      - 去掉 bump 行中的 `-bm <val>`，并记录强度
      - 丢弃 Blender 无用/不支持的 map_Ka / map_Kf / Tf / illum
      - 保留常用的 Kd/Ks/Ni/d，map_Kd/map_Ks/map_d/bump

    返回:
      fixed_mtl_path: Path
      bump_info: dict[material_name] = (texture_rel_path, strength_float)
    """
    if not mtl_path.exists():
        return mtl_path, {}

    lines_out = []
    bump_info = {}
    current_mat = None

    raw = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in raw:
        L = line.strip()
        # 记录当前材质名
        if L.lower().startswith("newmtl"):
            current_mat = L.split(maxsplit=1)[1].strip()
            lines_out.append(line)
            continue

        # 丢弃不需要/不支持的字段
        if L.startswith(("map_Ka", "map_Kf", "Tf", "illum")):
            continue

        # 统一路径分隔符
        if "map_" in L or L.startswith("bump"):
            line = line.replace("\\", "/")

        # bump: 取出强度，清掉 -bm 参数
        if L.startswith("bump"):
            parts = line.split()
            tex = None; strength = 1.0
            if len(parts) >= 2:
                tex = parts[1]
            if "-bm" in parts:
                try:
                    idx = parts.index("-bm")
                    if idx + 1 < len(parts):
                        strength = float(parts[idx+1])
                except Exception:
                    pass
                # 去掉 -bm 及其数值
                line = " ".join(parts[:idx]) + "\n"
            if current_mat and tex:
                bump_info[current_mat] = (tex, strength)
            lines_out.append(line)
            continue

        # 其余行原样保留（含 Kd/Ks/Ni/d、map_Kd/map_Ks/map_d 等）
        lines_out.append(line)

    fixed_mtl = mtl_path.with_name(mtl_path.stem + "_fixed.mtl")
    fixed_mtl.write_text("\n".join(lines_out), encoding="utf-8")
    return fixed_mtl, bump_info

def ensure_principled(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    out  = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
    if not out:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    if not bsdf:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return bsdf, out

def fix_material_slots(obj, search_dirs, bump_info=None):
    """
    - 若材质无贴图，尝试保留现状
    - 若 bump_info 中有该材质，则创建 Normal Map 节点并设置 strength
    """
    from pathlib import Path
    for mat in obj.data.materials or []:
        if not mat: 
            continue
        bsdf, out = ensure_principled(mat)
        nt = mat.node_tree

        # bump/normal
        if bump_info and mat.name in bump_info:
            tex_rel, strength = bump_info[mat.name]
            tex_path = None
            for d in search_dirs:
                p = Path(d) / tex_rel
                if p.exists():
                    tex_path = p; break
            if tex_path:
                img = bpy.data.images.load(str(tex_path), check_existing=True)
                tex_node = nt.nodes.new("ShaderNodeTexImage")
                tex_node.image = img
                nm_node = nt.nodes.new("ShaderNodeNormalMap")
                nm_node.inputs["Strength"].default_value = float(strength)
                nt.links.new(tex_node.outputs["Color"], nm_node.inputs["Color"])
                nt.links.new(nm_node.outputs["Normal"], bsdf.inputs["Normal"])

def check_materials(obj):
    """检查一个合并后的 object 是否正确挂上贴图"""
    if obj.type != "MESH":
        print(f"[Skip] {obj.name} is not a mesh")
        return
    print(f"Object: {obj.name}, material slots = {len(obj.material_slots)}")
    for i, slot in enumerate(obj.material_slots):
        mat = slot.material
        if not mat:
            print(f"  Slot {i}: [!] No material")
            continue
        print(f"  Slot {i}: Material = {mat.name}")
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            tex_nodes = [n for n in nodes if n.type == "TEX_IMAGE"]
            if not tex_nodes:
                print("    [!] No Image Texture node found")
            for tn in tex_nodes:
                img = tn.image
                if img:
                    print(f"    Image: {img.filepath} size={img.size}")
                else:
                    print("    [!] TEX_IMAGE node exists but no image loaded")
        else:
            print("    [!] Material not using nodes")

def is_orthonormal(R, eps=1e-5):
    R = np.asarray(R, float).reshape(3,3)
    return (np.allclose(R.T @ R, np.eye(3), atol=eps) and np.linalg.det(R) > 0)

def orthonormalize_R_if_needed(R, eps=1e-5):
    R = np.asarray(R, float).reshape(3,3)
    if is_orthonormal(R, eps):
        return R  # 保留你的朝向
    # 否则才做SVD投影
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:
        U[:, -1] *= -1.0
        R_orth = U @ Vt
    return R_orth

def to_M4_from_R(R):
    R = np.asarray(R, dtype=float).reshape(3,3)
    return Matrix(((R[0,0], R[0,1], R[0,2], 0.0),
                   (R[1,0], R[1,1], R[1,2], 0.0),
                   (R[2,0], R[2,1], R[2,2], 0.0),
                   (0.0,     0.0,    0.0,   1.0)))
from mathutils import Matrix

def apply_target_orientation_with_temp_root(objs, R_3x3):
    """
    用一个临时空物体作为父节点，统一对一组对象施加朝向 R（3x3）。
    - R 会先作用于 root，再传递给所有子对象
    - 后续烘焙、清洗、join 都不会改变这次朝向
    返回：(root_obj)
    """
    import bpy
    # 建 root
    root = bpy.data.objects.new("OXL_Root", None)
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = Matrix.Identity(4)

    # 挂到 root
    for o in objs:
        try:
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()
        except Exception:
            pass

    # 施加 R（左乘）
    R4 = Matrix((
        (R_3x3[0][0], R_3x3[0][1], R_3x3[0][2], 0.0),
        (R_3x3[1][0], R_3x3[1][1], R_3x3[1][2], 0.0),
        (R_3x3[2][0], R_3x3[2][1], R_3x3[2][2], 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    root.matrix_world = R4 @ root.matrix_world
    return root

def align_and_rotate(obj, R):
    # 先找出物体局部Z方向
    z_local = obj.matrix_world.to_quaternion() @ Vector((0,0,1))
    z_local.normalize()
    z_world = Vector((0,0,1))

    # 旋转四元数：把 local Z 对齐 world Z
    axis = z_local.cross(z_world)
    if axis.length > 1e-6:
        angle = z_local.angle(z_world)
        align_q = axis.normalized().rotation_difference(z_local).inverted()
        obj.rotation_euler = align_q.to_euler()
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # 再应用 canonical 矩阵 R
    M = Matrix(R.tolist() + [[0,0,0]])  # 转成4x4
    M.resize_4x4()
    obj.matrix_world = M @ obj.matrix_world

# def join_imported_meshes_only(objs):
#     import bpy
#     meshes = [o for o in objs if o and o.type=='MESH']
#     if not meshes: return None
#     bpy.ops.object.select_all(action='DESELECT')
#     for o in meshes: o.select_set(True)
#     bpy.context.view_layer.objects.active = meshes[0]
#     bpy.ops.object.join()
#     return bpy.context.view_layer.objects.active

def join_imported_meshes(objs):
    import bpy
    # 只保留 MESH
    meshes = [o for o in objs if o.type == 'MESH']
    if not meshes:
        return None
    # 防止多用户冲突，先断开共享
    for m in meshes:
        try:
            m.select_set(True)
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)
        except:
            pass
    # 统一选择并设置 active
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        try:
            bpy.ops.object.join()  # join 后只剩下 active 对象
            joined = bpy.context.active_object
            return joined
        except Exception as e:
            print(f"[WARN] join failed: {e}")
    return bpy.context.active_object  # 返回合并后的整体

# ========== 几何：边界环与“空心+开口”判定 ==========
def world_boundary_loops(obj):
    """
    返回边界环列表：每个元素为 Nx3 numpy 坐标（世界坐标）
    """
    deps = bpy.context.evaluated_depsgraph_get()
    me_eval = obj.evaluated_get(deps).to_mesh()
    bm = bmesh.new(); bm.from_mesh(me_eval); bm.verts.ensure_lookup_table(); bm.edges.ensure_lookup_table()
    # 找边界边
    bdry_edges = [e for e in bm.edges if len(e.link_faces)==1]
    if not bdry_edges:
        bm.free(); obj.to_mesh_clear(); return []
    # 按顶点邻接在边界子图上组环
    # 建邻接
    adj = {}
    for e in bdry_edges:
        v0,v1 = e.verts
        adj.setdefault(v0.index, set()).add(v1.index)
        adj.setdefault(v1.index, set()).add(v0.index)

    visited = set()
    loops = []

    for seed_e in bdry_edges:
        v_start = seed_e.verts[0].index
        if v_start in visited: continue
        # 走一圈
        loop_idx = []
        current = v_start
        prev = None
        while True:
            loop_idx.append(current)
            visited.add(current)
            nbrs = list(adj.get(current, []))
            # 选下一个（排除来自的点）
            nxt = None
            for nb in nbrs:
                if nb != prev:
                    nxt = nb; break
            if nxt is None: break
            prev, current = current, nxt
            if current == v_start:
                break
        if len(loop_idx) >= 3:
            # 转为世界坐标
            coords = [obj.matrix_world @ bm.verts[i].co for i in loop_idx]
            loops.append(np.array([[c.x, c.y, c.z] for c in coords], dtype=float))

    bm.free(); obj.to_mesh_clear()
    return loops

def fit_plane(points):
    """
    points: Nx3
    返回 (center, normal(unit), U, V) 其中 U,V 为面内正交基
    """
    P = points - points.mean(axis=0, keepdims=True)
    _,_,vh = np.linalg.svd(P, full_matrices=False)
    n = vh[-1]; n = n / (np.linalg.norm(n)+1e-12)
    U = vh[0]; V = vh[1]
    return points.mean(axis=0), n, U, V

def loop_area_in_plane(points):
    """
    将环投影到拟合平面(U,V)后，估算多边形面积（绝对值）
    """
    c, n, U, V = fit_plane(points)
    UV = np.c_[(points - c)@U, (points - c)@V]
    # 多边形面积
    x,y = UV[:,0], UV[:,1]
    area = 0.5 * abs(np.sum(x*np.roll(y,-1) - y*np.roll(x,-1)))
    return area, (c,n,U,V)

def is_hollow_with_opening(obj, dbg=False,
                           plane_normal_tol_deg=15.0,
                           plane_gap_rel_max=0.12,
                           thickness_rel_min=0.01):
    """
    经验性判定：
      - 存在 >=2 个“近共面”的边界环（通常对应杯口内外沿）
      - 两环在该平面内的“平均半径差/对角线” >= thickness_rel_min（有壁厚）
    其中相对度量的归一尺度用整体 AABB 对角线。
    """
    loops = world_boundary_loops(obj)
    if dbg: print(f"  [DBG] boundary_loops={len(loops)}")
    if len(loops) < 2:
        return False, "no_or_single_boundary_loop"

    # 全局尺度
    bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    xs=[v.x for v in bb]; ys=[v.y for v in bb]; zs=[v.z for v in bb]
    diag = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2 + (max(zs)-min(zs))**2)
    if diag <= 1e-9:
        return False, "degenerate_bbox"

    # 计算每个环的平面与面积
    infos = []
    for L in loops:
        area, (c,n,U,V) = loop_area_in_plane(L)
        infos.append(dict(points=L, center=c, normal=n, U=U, V=V, area=area))

    # 以面积降序尝试两两配对（找最可能的“杯口内外沿”）
    infos.sort(key=lambda d: -d["area"])

    rad = math.radians(plane_normal_tol_deg)
    cos_tol = math.cos(rad)
    for i in range(len(infos)):
        for j in range(i+1, len(infos)):
            A, B = infos[i], infos[j]
            # 法向近似平行（共面/近共面）
            if abs(np.dot(A["normal"], B["normal"])) < cos_tol:
                continue
            # 平面间距（沿法向）
            gap = abs(np.dot(A["center"] - B["center"], A["normal"]))
            if gap/diag > plane_gap_rel_max:
                continue

            # 在 A 的(U,V)平面计算“平均半径”，近似厚度=外环半径-内环半径
            def mean_radius(info, ref):
                UV = np.c_[(info["points"]-ref["center"])@ref["U"],
                           (info["points"]-ref["center"])@ref["V"]]
                r = np.sqrt(UV[:,0]**2 + UV[:,1]**2)
                return float(np.mean(r))
            rA = mean_radius(A, A); rB = mean_radius(B, A)
            thick = abs(rA - rB)
            if (thick/diag) >= thickness_rel_min:
                # 找到一对“近共面双环 + 非零壁厚” → 视为空心且有开口
                return True, "hollow_open_ok"

    return False, "no_coplanar_loop_pair_with_thickness"
######
def realize_and_make_single_user():
    # 1) 实体化所有实例（Dupli/Collection Instance 等）
    instancers = [o for o in bpy.context.scene.objects if getattr(o, "is_instancer", False) or o.instance_type != 'NONE']
    if instancers:
        bpy.ops.object.select_all(action='DESELECT')
        for o in instancers:
            o.select_set(True)
        bpy.context.view_layer.objects.active = instancers[0]
        bpy.ops.object.duplicates_make_real()

    # 2) 断开共享的 Mesh/材质（让每个对象拥有自己的数据）
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if meshes:
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        # 新旧 API 兼容
        try:
            bpy.ops.object.make_single_user(object=True, obdata=True, material=True,
                                            animation=False, obdata_animation=False)
        except TypeError:
            bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True, material=True,
                                            animation=False, obdata_animation=False)
        # 保底：仍共享时强制拷贝 data
        for o in meshes:
            if getattr(o.data, "users", 1) > 1:
                o.data = o.data.copy()

def recenter_world(obj):
    """把物体在世界坐标的几何中心搬到世界原点（清掉 world 平移，保留朝向/比例）"""
    bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    center_world = sum(bb, Vector()) / 8.0
    obj.matrix_world = Matrix.Translation(-center_world) @ obj.matrix_world

# def normalize_object(obj, target_size=1.0):
#     # 以包围盒中心为原点，把最大边归一到 target
#     bpy.context.view_layer.objects.active = obj
#     bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
#     dims = obj.dimensions
#     max_dim = max(dims.x, dims.y, dims.z)
#     s = (target_size / max_dim) if max_dim > 0 else 1.0
#     obj.scale = (obj.scale.x * s, obj.scale.y * s, obj.scale.z * s)
#     bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
#     obj.location = (0.0, 0.0, 0.0)
import bpy
from mathutils import Vector, Matrix

# ---------- helpers ----------
def _select_only(obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

def _world_aabb(obj, mute_constraints=False):
    """返回包含修改器的 世界AABB (min_v, max_v)"""
    if mute_constraints and getattr(obj, "constraints", None):
        saved = [(c, c.mute) for c in obj.constraints]
        for c, _ in saved: c.mute = True

    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    # 优先 bound_box（快），必要时 to_mesh（稳）
    bb = getattr(obj_eval, "bound_box", None)
    if bb:
        pts = [obj_eval.matrix_world @ Vector(c) for c in bb]
    else:
        me = obj_eval.to_mesh()
        try:
            pts = [obj_eval.matrix_world @ v.co for v in me.vertices]
        finally:
            obj_eval.to_mesh_clear()
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return min_v, max_v

def bake_world_geometry(obj) -> bpy.types.Object:
    import bpy
    from mathutils import Matrix
    dg = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(dg)
    me = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
    try:
        # 把世界坐标写进一份新 mesh（顶点直接乘以 matrix_world）
        new_me = bpy.data.meshes.new(obj.name + "_baked")
        new_me.from_mesh(me)
        mw = obj_eval.matrix_world.copy()
        new_me.transform(mw)  # 关键：把世界变换烘进顶点
        new_obj = obj
        new_obj.data = new_me
        new_obj.matrix_world = Matrix.Identity(4)  # 关键：对象变换清为单位
        return new_obj
    finally:
        obj_eval.to_mesh_clear()

# def _world_aabb(obj, mute_constraints=False):
#     """返回包含修改器的 世界AABB (min_v, max_v)，用 evaluated mesh 精确计算。"""
#     if mute_constraints and getattr(obj, "constraints", None):
#         saved = [(c, c.mute) for c in obj.constraints]
#         for c, _ in saved: c.mute = True
#     dg = bpy.context.evaluated_depsgraph_get()
#     obj_eval = obj.evaluated_get(dg)
#     me = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=dg)
#     try:
#         if not me or len(me.vertices) == 0:
#             from mathutils import Vector
#             z = Vector((0,0,0))
#             return z, z
#         pts = [obj_eval.matrix_world @ v.co for v in me.vertices]
#     finally:
#         if me:
#             obj_eval.to_mesh_clear()
#     from mathutils import Vector
#     min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
#     max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
#     return min_v, max_v

def _recenter_and_normalize_world(obj, target_size=1.0, mute_constraints=True):
    """
    把世界AABB中心搬到(0,0,0)，并把世界最大边缩放到 target_size。
    不改 origin，不直接改 location；通过左乘世界矩阵完成。
    """
    # 暂时静音约束，避免再次“拉走”
    saved = None
    if mute_constraints and getattr(obj, "constraints", None):
        saved = [(c, c.mute) for c in obj.constraints]
        for c, _ in saved: c.mute = True

    try:
        min_v, max_v = _world_aabb(obj, mute_constraints=False)
        ext = (max_v - min_v)
        max_dim = max(ext.x, ext.y, ext.z)
        if not (max_dim and max_dim > 1e-12):
            return False  # 空/退化
        center = (min_v + max_v) * 0.5
        s = (target_size / max_dim)

        T = Matrix.Translation(-center)          # 先平移到原点
        S = Matrix.Diagonal((s, s, s, 1.0))      # 再围绕原点等比缩放
        obj.matrix_world = S @ T @ obj.matrix_world
        return True
    finally:
        if saved:
            for c, was in saved: c.mute = was

def normalize_unit_pipeline(main_obj, target_size: float):
    """
    统一的“单体化 → 转Mesh → 世界居中与归一化 → 应用旋转/缩放(不应用location)”流程
    """
    _select_only(main_obj)
    # bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)

    # 实体化实例
    if getattr(main_obj, "instance_type", 'NONE') != 'NONE':
        bpy.ops.object.duplicates_make_real()
        meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
        if meshes:
            main_obj = meshes[0]
            _select_only(main_obj)

    # 转成 Mesh（若已是 Mesh 则跳过）
    if main_obj.type != 'MESH':
        bpy.ops.object.convert(target='MESH', keep_original=False)
        main_obj = bpy.context.view_layer.objects.active

    # === A) 先检查是否为空/退化（允许 >1，禁止 0/NaN）===
    min_v, max_v = _world_aabb(main_obj, mute_constraints=False)
    ext = (max_v - min_v)
    max_dim_world = max(ext.x, ext.y, ext.z)
    if not (max_dim_world and max_dim_world > 1e-8):
        return False, "invalid_size"

    # === B) 做“世界居中 + 归一化” ===
    ok = _recenter_and_normalize_world(main_obj, target_size=target_size, mute_constraints=True)
    if not ok:
        return False, "invalid_size"

    # === C) 应用旋转/缩放（不应用 location）===
    _select_only(main_obj)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    bpy.context.view_layer.update()

    # === D) 归一化后再测一次 AABB，最大边 ≤ 1 ===
    scale_error = 1e-2 # not effect the later nocs calculation
    min_v2, max_v2 = _world_aabb(main_obj, mute_constraints=False)
    L = max((max_v2 - min_v2).x, (max_v2 - min_v2).y, (max_v2 - min_v2).z)
    if L > 1.0 + 1e-5:
        corr = 1.0 / L
        main_obj.matrix_world = Matrix.Diagonal((corr, corr, corr, 1.0)) @ main_obj.matrix_world
        _select_only(main_obj)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        # 再测一次做最终断言（严格保障 NOCS）
        min3, max3 = _world_aabb(main_obj)
        L1 = max((max3 - min3).x, (max3 - min3).y, (max3 - min3).z)
        assert L1 <= 1.0 + scale_error, f"normalized L={L1} > 1" # 1e-3 as error
    # print(f"[align] normalized L={L} (target_size={target_size})")
    # 收敛 location 的极小残差（非必要）
    eps = 1e-9
    loc = main_obj.location
    main_obj.location = Vector((
        0.0 if abs(loc.x) < eps else loc.x,
        0.0 if abs(loc.y) < eps else loc.y,
        0.0 if abs(loc.z) < eps else loc.z,
    ))
    return True, "-"

# def normalize_object(objs, target_size=1.0, outlier_ratio=10.0, bake_scale=True, recenter='bbox', drop_to_ground=True):
#     """
#     用评估后的世界几何做归一化，并排除离群网格。
#     - objs: 传入要归一化的一组对象（常用：已选对象）
#     - target_size: 归一化后最大边（米）
#     - outlier_ratio: > 中位数 * ratio 视作离群，不参与量测
#     - bake_scale: 是否将缩放烘焙进几何（transform_apply(scale=True)）
#     返回: (scale_factor, prev_max_dim)
#     """
#     import bpy
#     mesh_objs = _gather_renderable_meshes(objs or bpy.context.selected_objects)
#     if not mesh_objs:
#         print("[align] no mesh objects to normalize; skip.")
#         return 1.0, 0.0

#     dg = bpy.context.evaluated_depsgraph_get()

#     # 第一次量测：包含所有对象
#     max_dim_all, per_obj_dims = _world_bbox_max_dim(mesh_objs, dg)

#     # 离群过滤：用各对象 max_dim 的中位数做阈值
#     per_dims = sorted([max(d.x, d.y, d.z) for _, d in per_obj_dims if d.length > 0.0])
#     if per_dims:
#         mid = per_dims[len(per_dims)//2]
#         cutoff = mid * float(outlier_ratio) if mid > 0 else float('inf')
#         mesh_core = [ob for (ob, d) in per_obj_dims if 0.0 < max(d.x, d.y, d.z) <= cutoff]
#     else:
#         mesh_core = mesh_objs

#     # 第二次量测：仅核心对象
#     max_dim_core, _ = _world_bbox_max_dim(mesh_core, dg)
#     prev_max = max_dim_core if max_dim_core > 0 else max_dim_all
#     if prev_max <= 0:
#         print("[align] invalid max_dim; skip scaling.")
#         return 1.0, 0.0

#     s = float(target_size) / float(prev_max)

#     # 找一个根来承载缩放；没有唯一根就建一个空物体
#     roots = [o for o in mesh_objs if not o.parent]
#     if len(roots) == 1:
#         align_root = roots[0]
#     else:
#         align_root = bpy.data.objects.new("ALIGN_ROOT", None)
#         bpy.context.scene.collection.objects.link(align_root)
#         for o in mesh_objs:
#             o.parent = align_root

#     align_root.scale = (align_root.scale[0]*s, align_root.scale[1]*s, align_root.scale[2]*s)
#     if bake_scale:
#         bpy.context.view_layer.objects.active = align_root
#         with contextlib.suppress(Exception):
#             bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
#     if recenter != 'none' or drop_to_ground:
#         import mathutils
#         dg2 = bpy.context.evaluated_depsgraph_get()
#         # 注意：用“缩放后”的几何重新量测
#         minv, maxv = _world_bbox_minmax(mesh_objs, dg2)

#         delta = mathutils.Vector((0.0, 0.0, 0.0))
#         if recenter in ('bbox', 'center'):
#             center = 0.5 * (minv + maxv)
#             delta -= center  # 把包围盒中心移到 (0,0,0)

#         if drop_to_ground:
#             # 先按 recenter 计算后的最小 Z 再修正，使底面到 Z=0
#             new_min_z = minv.z + delta.z
#             delta.z -= new_min_z

#         # 平移施加到对齐根（上面我们已经用 align_root 做缩放）
#         align_root.location = (align_root.location[0] + delta.x,
#                                align_root.location[1] + delta.y,
#                                align_root.location[2] + delta.z)
#         print(f"[align] recentered by delta={tuple(round(x,6) for x in delta)} "
#               f"(recenter={recenter}, drop_to_ground={drop_to_ground})")

#     print(f"[align] normalized scale: target={target_size:.4f}m, prev_max={prev_max:.6f} -> scale={s:.6f}")
#     return s, prev_max

def set_cycles(prefer_gpu=True):
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = 64
    scn.cycles.use_denoising = True
    scn.cycles.device = 'CPU'
    if not prefer_gpu:
        return
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.refresh_devices()
        # 优先 OPTIX，再 CUDA
        prefs.compute_device_type = 'OPTIX' if any(d.type=='OPTIX' for d in prefs.devices) else 'CUDA'
        for d in prefs.devices: d.use = True
        scn.cycles.device = 'GPU'
    except Exception:
        pass

def add_lights_and_world():
    world = bpy.data.worlds.new("World") if not bpy.context.scene.world else bpy.context.scene.world
    world.use_nodes = True
    bpy.context.scene.world = world
    # key sun
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new(name="Sun", type='SUN'))
    sun.data.energy = 3.0
    sun.location = (6, -6, 8)
    bpy.context.collection.objects.link(sun)
    # fill area
    area = bpy.data.objects.new("Area", bpy.data.lights.new(name="Area", type='AREA'))
    area.data.energy = 1200
    area.data.size = 6.0
    area.location = (-6, 6, 6)
    bpy.context.collection.objects.link(area)

def _world_bounds_of(obj):
    # obj.bound_box 是局部 8 顶点，乘 world 矩阵得到世界坐标
    ws = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    minv = Vector((min(v.x for v in ws), min(v.y for v in ws), min(v.z for v in ws)))
    maxv = Vector((max(v.x for v in ws), max(v.y for v in ws), max(v.z for v in ws)))
    return minv, maxv

def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

def add_world_axes(size=1.0, thickness=0.05):
    def make_arrow(axis, color, loc, rot):
        bpy.ops.mesh.primitive_cylinder_add(
            radius=thickness, depth=size, location=loc, rotation=rot
        )
        arrow = bpy.context.active_object
        mat = bpy.data.materials.new(name=f"Axis_{axis}")
        mat.diffuse_color = (*color, 1.0)
        arrow.data.materials.append(mat)
        return arrow

    arrows = []
    # X axis
    arrows.append(make_arrow("X", (1,0,0), (size/2,0,0), (0,0,math.radians(90))))
    # Y axis
    arrows.append(make_arrow("Y", (0,1,0), (0,size/2,0), (math.radians(90),0,0)))
    # Z axis
    arrows.append(make_arrow("Z", (0,0,1), (0,0,size/2), (0,0,0)))
    
    axes = bpy.data.collections.new("WorldAxes")
    bpy.context.scene.collection.children.link(axes)
    for a in arrows:
        bpy.data.collections["WorldAxes"].objects.link(a)
        bpy.context.scene.collection.objects.unlink(a)
    return arrows

def place_camera_fit(minv, maxv, out_path, res=1600, fov_deg=75.0, base_bias=10.0, k=0.4, N=6, topdown=False, verbose=False, engine='CYCLES',
                                ):
    scn = bpy.context.scene
    # scn.render.engine = engine

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", cam_data)
    # cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    scn.camera = cam
    # 计算中心和跨度
    minv = Vector(minv); maxv = Vector(maxv)
    center = (minv + maxv) * 0.5
    span = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z)
    if verbose:
        print(f"[CAM DEBUG] all_min={minv}, all_max={maxv}, center={center}, span={span}")
    xy_span = max((maxv - minv).x, (maxv - minv).y)
    z_span  = (maxv - minv).z
    # 相机放在对角线方向
    fov = math.radians(fov_deg) # larger fov
    dist = (span * 0.6) / math.tan(fov/2.0) * 1.2 # 0.6 留一点边距
    # adjust camera height based on grid numbers
    up_bias = base_bias + k * (N - 3)
    if topdown:
        # cam.location = center + Vector((0, 0, 5)) # why 2 is so small span*2
        # cam.rotation_euler = (math.radians(90), 0, 0)
        # # cam.data.type = 'ORTHO'
        # # cam.data.ortho_scale = span * 1.2
        cam.data.type = 'ORTHO'
        # 画幅大小：覆盖 XY 跨度再留点边（1.15~1.3）
        if N>=10.0 and N<20.0:
            cam.data.ortho_scale = xy_span * (N+5) if xy_span > 0 else 1.0
        elif N<10.0: 
            cam.data.ortho_scale = xy_span * 13.0 if xy_span > 0 else 1.0
        else: 
            cam.data.ortho_scale = xy_span * 2* N if xy_span > 0 else 1.0
        z_height = max(1.0, z_span) * 12.5 + N * 0.3    
        # 相机位置：在中心正上方，抬到足够高；高度与可见范围无关（正交），
        # 但要避开 clip_start；设置 clip_end 足够大
        
        cam.location = center + Vector((0, 0, z_height))
        # 朝向中心（不要手写欧拉，直接“look at”最稳）
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
        # 裁剪面
        cam.data.clip_start = 0.001
        cam.data.clip_end   = max(20.0, (xy_span + z_span) * 100.0)
    else:
        # 斜视角模式
        cam_dir = Vector((1.2, -1.2, up_bias)).normalized() # maybe normalization makes the fov too small
        cam.location = center + cam_dir * dist
        cam.data.angle = fov
        direction = center - cam.location
        cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    look_at(cam, center)
    # 渲染设置
    scn.render.resolution_x = res
    scn.render.resolution_y = int(res * 0.75)
    scn.render.film_transparent = True
    scn.render.filepath = str(out_path)
    if verbose:
        print(f"[CAM] type={cam.data.type} ortho_scale={getattr(cam.data,'ortho_scale',None)} "
            f"loc={tuple(cam.location)} span(xy,z)=({xy_span:.3f},{z_span:.3f})")
        
def center_symmetric_grid_offsets(count, N, cell):
    # 以网格中心为原点对称排布：列/行坐标减去 (N-1)/2
    rows = math.ceil(count / N)
    offsets = []
    for i in range(count):
        r, c = divmod(i, N)
        cx = (c - (N-1)/2) * cell
        cy = (((rows-1)/2) - r) * cell  # Y朝上
        offsets.append((cx, cy, 0.0))
    return offsets

def clear_scene_hard():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # 额外做孤儿清理，确保数据块也被清掉
    try:
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass

def make_mesh_single_user(obj):
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True, material=True)
    except TypeError:
        bpy.ops.object.make_single_user(object=True, obdata=True, material=True)
    # 把修改器烘焙到 evaluated mesh，避免布尔/阵列造成的奇怪导出
    deps = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(deps)
    new_mesh = bpy.data.meshes.new_from_object(eval_obj)
    obj.data = new_mesh

def fix_normals(obj):
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

def export_only_selected_glb(obj_or_objs, out_path):
    import bpy, os
    # 统一成列表
    objs = obj_or_objs if isinstance(obj_or_objs, (list, tuple)) else [obj_or_objs]

    # 精确选择要导出的对象
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]

    # 预先把外链图片打包进 .blend（Exporter 会用打包数据，最稳）
    try:
        bpy.ops.file.pack_all()
    except Exception:
        pass

    # 组装参数（兼容不同 Blender 版本）
    op = bpy.ops.export_scene.gltf
    kwargs = dict(
        filepath=str(out_path),
        export_format='GLB',   # 关键：GLB 封装（图片内嵌）
        use_selection=True,
        export_apply=True,
        export_image_format='AUTO',  # 让导出器根据原格式/alpha 自行决定
        export_unused_images=True,   # ← 新增：未被材质使用的图片也写入 GLB
        # 可按需：'use_visible': True
    )
    try:
        props = op.get_rna_type().properties
        for k in list(kwargs.keys()):
            if k not in props:
                kwargs.pop(k)
        op(**kwargs)
    except Exception as e:
        raise RuntimeError(f"GLB export failed: {e}")

    # 简单校验 GLB 头与尺寸
    try:
        with open(out_path, 'rb') as f:
            magic = f.read(4)
        size = os.path.getsize(out_path)
        print(f"[GLB] exported: {out_path.name} size={size} magic={magic}")
    except Exception:
        pass

    bpy.ops.object.select_all(action='DESELECT')


import bpy, os, shutil

def _save_or_copy_image(img: bpy.types.Image, dst_dir: Path) -> Path | None:
    """
    把材质中用到的图片保存到 dst_dir：
      - 若是打包图像：另存为 PNG；
      - 若有文件路径：拷贝到 dst_dir；
    返回目标路径（相对 dst_dir 的文件）。
    """
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        # 尝试用原始文件名
        base = Path(img.filepath_raw).name if img.filepath_raw else (img.name + ".png")
        if not base:
            base = img.name + ".png"
        # 去掉奇怪字符
        base = base.replace("\\", "/").split("/")[-1]
        tgt = dst_dir / base

        if img.packed_file is not None:
            # 打包的直接另存为 PNG（最稳）
            tmp_format = img.file_format
            img.file_format = 'PNG'
            img.save(filepath=str(tgt))
            img.file_format = tmp_format
        else:
            src = Path(bpy.path.abspath(img.filepath_raw)) if img.filepath_raw else None
            if src and src.exists():
                if str(src.resolve()) != str(tgt.resolve()):
                    shutil.copy2(str(src), str(tgt))
            else:
                # 没路径也没打包，就忽略
                return None
        return tgt
    except Exception:
        return None

# def collect_and_dump_textures(obj: bpy.types.Object, textures_dir: Path) -> dict[str, str]:
#     """
#     遍历对象所有材质节点，把出现的 Image 另存/拷贝到 textures_dir。
#     返回 {原文件名: 相对 textures/ 的文件名} 的映射，便于后续替换 MTL。
#     """
#     mapping = {}
#     for mat in obj.data.materials or []:
#         if not mat or not getattr(mat, "use_nodes", False) or not mat.node_tree:
#             continue
#         for n in mat.node_tree.nodes:
#             if n.type == "TEX_IMAGE" and n.image:
#                 dst = _save_or_copy_image(n.image, textures_dir)
#                 if dst:
#                     mapping[Path(n.image.filepath_raw).name if n.image.filepath_raw else n.image.name] = dst.name
#     return mapping
# —— 收集纹理到 textures/ 并将文件名安全化，返回“原名→安全名”映射（大小写无关）
def collect_and_dump_textures(
    bpy_obj, textures_dir: Path, rename: bool = True
) -> Dict[str, str]:
    """
    从 Blender 节点/材质中收集贴图，复制到 textures/；如果 rename=True，同步改名为安全名。
    返回：orig_basename(casefold) -> safe_basename
    """
    textures_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, str] = {}

    import bpy  # 在 Blender 中运行

    def save_or_copy_image(img, dst_dir: Path) -> Optional[Path]:
        # 尝试拿到原路径；没有就导出为 PNG
        raw = Path(bpy.path.abspath(img.filepath_raw or img.filepath)) if (img.filepath_raw or img.filepath) else None
        if raw and raw.exists():
            src = raw
            base = src.name
            safe = sanitize_basename(base) if rename else base
            dst = dst_dir / safe
            if str(src.resolve()) != str(dst.resolve()):
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src), str(dst))
            return dst
        else:
            # 导出内嵌图像为 PNG
            safe = sanitize_basename((img.name or "image") + ".png") if rename else (img.name or "image") + ".png"
            dst = dst_dir / safe
            dst.parent.mkdir(parents=True, exist_ok=True)
            # 保存为 PNG（避免奇怪格式）
            try:
                # 临时设置并保存
                prev_fp = img.filepath
                img.filepath = str(dst)
                img.file_format = 'PNG'
                img.save()
                img.filepath = prev_fp
                return dst
            except Exception:
                return None

    # 遍历材质节点收集
    for mat in (bpy_obj.data.materials or []):
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if getattr(node, "type", "") == "TEX_IMAGE" and getattr(node, "image", None):
                dst = save_or_copy_image(node.image, textures_dir)
                if dst:
                    orig = Path(node.image.filepath_raw or node.image.filepath or node.image.name).name
                    mapping[_canon_key(orig)] = dst.name

    return mapping

def ensure_mtl_has_maps(obj, mtl_path: Path, textures_dir: Path):
    import bpy, os, shutil, re
    # 读取 MTL
    txt = mtl_path.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    textures_dir.mkdir(parents=True, exist_ok=True)

    # -------- Pass 0: 先统计每个 newmtl 块是否已有 map_ 行（按块判断，别整文件 early return）
    has_map = {}        # name -> bool
    cur = None
    for l in lines:
        s = l.strip()
        if s.lower().startswith("newmtl"):
            cur = s.split(None, 1)[1] if " " in s else ""
            has_map.setdefault(cur, False)
        elif cur and s.lower().startswith(("map_", "bump")):
            has_map[cur] = True

    # -------- 收集每个材质的贴图（沿用你原逻辑）
    def dump_image(img) -> str | None:
        if not img: return None
        bn = Path(bpy.path.basename(img.filepath_raw) or img.name).name
        if not bn.lower().endswith((".png",".jpg",".jpeg",".tga",".bmp",".exr",".tif",".tiff",".webp")):
            bn += ".png"
        dst = textures_dir / bn
        try:
            abspath = bpy.path.abspath(img.filepath) if img.filepath else ""
            if img.packed_file:
                img.save_render(str(dst))
            elif abspath and os.path.exists(abspath):
                if str(dst) != abspath:
                    shutil.copy2(abspath, dst)
            else:
                img.save_render(str(dst))
            return bn
        except Exception:
            return None

    mat_maps = {}  # name -> {"kd":..., "nm":..., "al":...}
    for mat in filter(None, getattr(obj.data, "materials", []) or []):
        kd = nm = al = None
        if getattr(mat, "use_nodes", False) and mat.node_tree:
            nt = mat.node_tree
            bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf:
                # Base Color
                link = next((l for l in nt.links if l.to_socket == bsdf.inputs.get("Base Color")), None)
                node = getattr(link, "from_node", None)
                if node and node.type == "TEX_IMAGE":
                    kd = dump_image(node.image)
                # Normal（优先 NORMAL_MAP → Color）
                link = next((l for l in nt.links if l.to_socket == bsdf.inputs.get("Normal")), None)
                node = getattr(link, "from_node", None)
                if node and node.type == "NORMAL_MAP":
                    link2 = next((l for l in nt.links if l.to_socket == node.inputs.get("Color")), None)
                    node2 = getattr(link2, "from_node", None)
                    if node2 and node2.type == "TEX_IMAGE":
                        nm = dump_image(node2.image)
                # Alpha（可选）
                link = next((l for l in nt.links if l.to_socket == bsdf.inputs.get("Alpha")), None)
                node = getattr(link, "from_node", None)
                if node and node.type == "TEX_IMAGE":
                    al = dump_image(node.image)
        if kd or nm or al:
            mat_maps[mat.name] = {"kd": kd, "nm": nm, "al": al}

    # 若完全没采到任何贴图，就不改
    if not mat_maps:
        return

    # -------- Pass 1：按块注入（修掉 newmtl 分支的 continue 导致的“永不到达”问题）
    out = []
    cur_name = None
    for line in lines:
        s = line.strip()
        if s.lower().startswith("newmtl"):
            cur_name = s.split(None, 1)[1] if " " in s else ""
            out.append(line)
            # 仅当该块目前没有 map_ 且我们能从节点树拿到图时，立刻补写
            if not has_map.get(cur_name, False) and cur_name in mat_maps:
                m = mat_maps[cur_name]
                def _q(p): return f'"{p}"' if " " in p else p
                # if m.get("kd"): out.append(f'map_Kd {_q("textures/"+m["kd"])}')
                # if m.get("nm"): out.append(f'map_Bump -bm 1 {_q("textures/"+m["nm"])}')
                # if m.get("al"): out.append(f'map_d {_q("textures/"+m["al"])}')
                # get away from the ""
                if m.get("kd"): out.append(f'map_Kd textures/{sanitize_basename(m["kd"])}')
                if m.get("nm"): out.append(f'map_Bump -bm 1 textures/{sanitize_basename(m["nm"])}')
                if m.get("al"): out.append(f'map_d textures/{sanitize_basename(m["al"])}')
            continue

        out.append(line)

    # 若 MTL 里根本没有任何 newmtl，但我们采到了贴图：补一个默认块
    if not has_map and not any(l.strip().lower().startswith("newmtl") for l in lines):
        # 选第一个有贴图的材质名
        name, m = next(iter(mat_maps.items()))
        out += [f"newmtl {name}"]
        def _q(p): return f'"{p}"' if " " in p else p
        if m.get("kd"): out.append(f'map_Kd {_q("textures/"+m["kd"])}')
        if m.get("nm"): out.append(f'map_Bump -bm 1 {_q("textures/"+m["nm"])}')
        if m.get("al"): out.append(f'map_d {_q("textures/"+m["al"])}')

    mtl_path.write_text("\n".join(out), encoding="utf-8")

# ======================== MTL/Texture 修复工具（可直接粘贴） ========================
from __future__ import annotations
import os, re, shlex, unicodedata, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# —— 规则：仅保留安全字符；空格与奇字符 → "_"；扩展名统一小写 ——
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_basename(name: str) -> str:
    """仅规范化 basename：空格/奇字符→'_'；扩展名小写；去首尾 '_'。"""
    name = unicodedata.normalize("NFC", os.path.basename(str(name)))
    stem, ext = os.path.splitext(name)
    stem = _SAFE_CHARS_RE.sub("_", stem).strip("_")
    return f"{stem}{ext.lower()}" if ext else stem

def _canon_key(s: str) -> str:
    """用于字典键（大小写无关匹配）的规范化：NFC + basename + casefold。"""
    return unicodedata.normalize("NFC", os.path.basename(str(s))).casefold()
# ===================== 导出专用 MTL 修复（可直接粘贴） =====================
from __future__ import annotations
import os, re, shlex, unicodedata, shutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# 仅保留安全字符；空格/非常规字符 -> "_"；扩展名统一小写
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")

def sanitize_basename(name: str) -> str:
    """仅规范化 basename：空格/奇字符→'_'；扩展名小写；去首尾 '_'。"""
    name = unicodedata.normalize("NFC", os.path.basename(str(name)))
    stem, ext = os.path.splitext(name)
    stem = _SAFE_CHARS_RE.sub("_", stem).strip("_")
    return f"{stem}{ext.lower()}" if ext else stem

def _canon_key(s: str) -> str:
    """生成大小写无关匹配用的键：NFC + basename + casefold。"""
    return unicodedata.normalize("NFC", os.path.basename(str(s))).casefold()

def build_available_textures_index(textures_dir: Path) -> Dict[str, Path]:
    """
    在 textures_dir 下建立大小写无关的可用文件索引：canon_key -> 真实 Path。
    """
    idx: Dict[str, Path] = {}
    if not textures_dir.exists():
        return idx
    for p in textures_dir.iterdir():
        if p.is_file():
            idx[_canon_key(p.name)] = p
    return idx

def ensure_texture_safe_name(p: Path) -> Path:
    """
    若 textures/ 下文件名包含空格/奇字符，则重命名为安全名；返回最终路径。
    """
    safe = sanitize_basename(p.name)
    if safe != p.name:
        q = p.with_name(safe)
        if not q.exists():
            p.rename(q)
        else:
            # 若已存在同名安全文件，选择覆盖或去重（这里选择覆盖为简单）
            q.unlink()
            p.rename(q)
        return q
    return p

def export_fix_mtl(
    mtl_path: Path,
    textures_dir: Path,
    write_fixed: bool = True,
    fixed_suffix: str = "_fixed",
    also_rewrite_obj: Optional[Path] = None,   # 若提供 OBJ 路径，则顺手改 mtllib 并产出 *_fixed.obj
    force_write_obj: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    导出阶段专用的 MTL 修复：
      1) 不删除 -bm、不删任何 map_*；最大限度保留原语义；
      2) 统一反斜杠为 '/'；
      3) 贴图路径一律写成相对路径：textures/<安全名>（无引号）；
      4) 把 textures/ 下的贴图文件名同步改为安全名（空格/奇字符→'_'，扩展名小写）；
      5) 大小写无关匹配以避免 Cat.png / cat.PNG 误判；
      6) 可选把 OBJ 的 mtllib 指向 *_fixed.mtl 并稳定产出 *_fixed.obj。

    返回：(最终 mtl 路径, 最终 obj 路径或 None)
    """
    if not mtl_path.exists():
        return mtl_path, None

    textures_dir.mkdir(parents=True, exist_ok=True)

    # 先把 textures/ 里已有文件改名为安全名，并建立索引
    idx: Dict[str, Path] = {}
    for p in list(textures_dir.iterdir()):
        if p.is_file():
            q = ensure_texture_safe_name(p)
            idx[_canon_key(q.name)] = q

    # 读取并重写 MTL
    KEYS = ("map_ka","map_kd","map_ks","map_ke","map_ns","map_d",
            "bump","map_bump","norm","disp","decal","refl")

    lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: List[str] = []

    for raw in lines:
        ln = raw
        s = ln.strip()
        if not s:
            out.append(ln); continue

        key = s.split(None, 1)[0].lower()
        if key in KEYS:
            # 统一反斜杠
            ln = ln.replace("\\", "/")

            # 提取选项与“最后一个非选项 token（文件路径）”
            rest = s[len(key):].strip()
            toks = shlex.split(rest, posix=True)
            if not toks:
                out.append(ln); continue

            opts, fname = [], None
            for t in toks:
                if t.startswith("-") and fname is None:
                    opts.append(t)
                else:
                    fname = t if fname is None else t
            if not fname:
                out.append(ln); continue

            old_bn = os.path.basename(fname)

            # 在索引中查找（大小写无关）；若没有，则以安全名落地（如果上游已复制，这里只规范名）
            canon = _canon_key(old_bn)
            if canon in idx:
                final_path = idx[canon]
            else:
                safe = sanitize_basename(old_bn)
                final_path = textures_dir / safe
                # 如果源文件还不在 textures/，这里不做跨目录搜索；导出阶段应已把贴图收集到 textures/
                # 若不存在，创建一个空占位避免 MTL 指向不存在（也可以选择跳过）
                if not final_path.exists():
                    final_path.touch()

                # 更新索引
                idx[_canon_key(final_path.name)] = final_path

            # 再保障一次安全名（万一是新 touch 的）
            final_path = ensure_texture_safe_name(final_path)

            # 写相对路径（无引号）
            new_rel = f"textures/{final_path.name}"
            ln = " ".join([key] + opts + [new_rel])
            out.append(ln)
        else:
            # 非贴图行：保留原始语义
            out.append(ln)

    # 输出 *_fixed.mtl（或原地覆盖）
    if write_fixed:
        mtl_fixed = mtl_path.with_name(mtl_path.stem + fixed_suffix + ".mtl")
        mtl_fixed.write_text("\n".join(out), encoding="utf-8")
    else:
        mtl_fixed = mtl_path
        mtl_path.write_text("\n".join(out), encoding="utf-8")

    obj_fixed: Optional[Path] = None
    if also_rewrite_obj:
        obj_fixed, _ = _rewrite_obj_mtllibs_export(
            also_rewrite_obj, {mtl_path.name: mtl_fixed.name}, force_write=force_write_obj, fixed_suffix=fixed_suffix
        )

    return mtl_fixed, obj_fixed

def _rewrite_obj_mtllibs_export(
    obj_path: Path,
    mtl_name_map: Dict[str, str],
    force_write: bool = True,
    fixed_suffix: str = "_fixed",
) -> Tuple[Path, bool]:
    """
    导出用的 mtllib 重写（健壮 & 稳定产出）：
      - 用 shlex 解析，支持一行多个 mtllib / 带空格 token；
      - 只替换文件名部分；
      - force_write=True 时无论比较结果都写 *_fixed.obj（或首次创建）。
    """
    if not obj_path.exists():
        return obj_path, False

    src = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, changed = [], False
    for ln in src:
        if ln.lstrip().lower().startswith("mtllib"):
            rest = ln.split(None, 1)[1] if " " in ln else ""
            toks = shlex.split(rest, posix=True)
            new_toks = []
            for tok in toks:
                base = os.path.basename(tok)
                rep = mtl_name_map.get(base, tok)
                if rep != tok:
                    changed = True
                new_toks.append(rep)
            out.append("mtllib " + " ".join(new_toks))
        else:
            out.append(ln)

    target = obj_path.with_name(obj_path.stem + fixed_suffix + ".obj")
    if force_write or changed or not target.exists():
        target.write_text("\n".join(out), encoding="utf-8")
        return target, True
    return (target if target.exists() else obj_path), False

def export_selected_as_obj_with_textures(obj: bpy.types.Object, out_dir: Path, name="model"):
    """
    将选中的对象导出为 OBJ/MTL，并把用到的贴图写入 textures/ 目录。
    最后把 MTL 里所有贴图路径改为相对 'textures/<file>'，并做最小修复（去 -bm，统一 /）。
    """
    import bpy
    # try:
    #     bpy.ops.file.pack_all()
    # except Exception:
    #     pass

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    textures_dir = out_dir / "textures"
    # 先收集/落盘贴图（以防 MTL 指向的路径不存在）
    export_path = out_dir / f"{name}.obj"
    base_dir = export_path.parent  # 以导出目录作为相对化基准

    # try:
    #     sanitize_and_relink_images(
    #         base_dir=str(out_dir),
    #         fbx_path=str(p) if p.suffix.lower()==".fbx" else None,   # p 是上层传进来的源文件
    #         extra_roots=[str((out_dir.parent / "raw"))]              # 通过相对关系推 raw；
    #     )
    # except Exception:
    #     pass
    tex_map = collect_and_dump_textures(bpy_obj=obj, textures_dir=textures_dir, rename=True)

    # 只导出所选
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    obj_path = out_dir / f"{name}.obj"

    # Blender 4.x: wm.obj_export；老版本：export_scene.obj
    op = None
    if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_export"):
        op = bpy.ops.wm.obj_export
    else:
        op = bpy.ops.export_scene.obj  # 兼容老版本

    # 组装参数：UV/法线/材质都导出；可按需三角化
    kwargs = dict(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_normals=True,
        export_uv=True,
        export_triangulated_mesh=False,  # 如果 PyTorch3D 需要三角面，可设 True
    )
    # 兼容不同版本的参数名
    props = op.get_rna_type().properties

    # Blender 4.x 新导出器
    if "forward_axis" in props:       # wm.obj_export 
        kwargs.update(forward_axis='Y', up_axis='Z') # Blender 轴向
        if "apply_transform" in props:
            kwargs["apply_transform"] = True  # 若有此参数，确保导出时应用物体矩阵

    try:
        props = op.get_rna_type().properties
        for k in list(kwargs.keys()):
            if k not in props:
                kwargs.pop(k)
        op(**kwargs)
    except Exception as e:
        raise RuntimeError(f"OBJ export failed: {e}")

    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.exists():
        return obj_path  # 没有材质也算成功

    # === NEW: 扫描 MTL 中的 map_* / bump 路径，把遗漏的贴图也拷到 textures/ ===
    import shlex, re
    try:
        lines_raw = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        lines_raw = []

    textures_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_paths(token: str):
        # 允许相对/绝对/带空格路径；给出几个候选位置
        p = Path(token)
        yield p
        yield (mtl_path.parent / token)
        # Blender 解析（若带 //）：
        try:
            import bpy
            absp = Path(bpy.path.abspath(token))
            yield absp
        except Exception:
            pass

    for raw in lines_raw:
        s = raw.strip().lower()
        if s.startswith("map_") or s.startswith("bump"):
            # 取最后一个非选项 token 当作路径
            parts = shlex.split(raw)  # 支持引号/空格
            if len(parts) >= 2:
                cand = parts[-1]
                # 已经收集过就略过
                base = Path(cand).name
                if base in tex_map:
                    continue
                # 尝试找到并复制
                found = False
                for cp in _candidate_paths(cand):
                    try:
                        if cp.exists() and cp.is_file():
                            dst = textures_dir / cp.name
                            if str(cp.resolve()) != str(dst.resolve()):
                                shutil.copy2(str(cp), str(dst))
                            tex_map[cp.name] = dst.name  # 记录 basename 映射
                            found = True
                            break
                    except Exception:
                        continue
                # （可选）提示未找到的条目
                if not found:
                    print(f"[warn] MTL map refers to missing file: {cand}")

    mtl_fixed, obj_fixed = export_fix_mtl(
        mtl_path=mtl_path,
        textures_dir=textures_dir,
        write_fixed=True,                 # 产出 *_fixed.mtl
        fixed_suffix="_fixed",
        also_rewrite_obj=obj_path,        # 可选：顺手改 mtllib，稳定产出 *_fixed.obj
        force_write_obj=True
    )

    # 3) 修好的内容覆盖回原 model.mtl，就执行：
    mtl_path.write_text(mtl_fixed.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        if mtl_fixed != mtl_path:
            mtl_fixed.unlink()  # 删除 *_fixed.mtl，避免双份
    except Exception:
        pass
    # # —— 最小修复：路径斜杠 & 去 -bm；同时把 map_* 的文件名改到 textures/ 子目录
    # fixed = fix_mtl_minimal(mtl_path)
    # ensure_mtl_has_maps(obj, fixed, textures_dir)
    # # 把 MTL 里的贴图路径替换为 'textures/<文件名>' （更稳健：支持空格文件名/任意选项）
    # lines = []
    # keys = set(tex_map.keys())  # 我们在导出前收集到的“原始 basename”（可能含空格）
    # for raw in fixed.read_text(encoding="utf-8", errors="ignore").splitlines():
    #     s_strip = raw.strip()
    #     low = s_strip.lower()
    #     if low.startswith("map_") or low.startswith("bump"):
    #         # 解析整行：head + opts + path
    #         tokens = shlex.split(raw)          # 支持引号/空格
    #         if len(tokens) >= 2:
    #             head = tokens[0]
    #             # 改成：只保留以 '-' 开头的真选项，其它一律丢弃
    #             raw_opts = tokens[1:-1]
    #             opts_tokens = [t for t in raw_opts if t.startswith('-')]
    #             orig_path = tokens[-1]
    #             bn = Path(orig_path).name

    #             # 你原来的“匹配已收集到的贴图名”的逻辑（可选）
    #             matched = next((k for k in keys if k and k in raw), None)
    #             target_name = tex_map.get(matched or bn, bn)

    #             new_path = f"textures/{target_name}"
    #             # 如果文件名里仍有空格，为稳妥可加引号（许多解析器也能接受）
    #             if " " in target_name:
    #                 new_path = f'"{new_path}"'

    #             out = " ".join([head] + opts_tokens + [new_path]).strip()
    #             lines.append(out)
    #             continue
    #     # 非贴图行或异常：原样保留
    #     lines.append(raw)

    # fixed.write_text("\n".join(lines), encoding="utf-8")
    # # —— 将修好的 fixed.mtl 覆盖回原始 model.mtl，避免 .obj 继续引用旧 mtl
    # try:
    #     fixed_text = fixed.read_text(encoding="utf-8")
    #     mtl_path.write_text(fixed_text, encoding="utf-8")
    #     # 可选：删除 fixed.mtl，避免双份文件引起困惑
    #     final_txt = mtl_path.read_text(encoding="utf-8", errors="ignore")
    #     if re.search(r"(^|[\s\"'])/|[A-Za-z]:/", final_txt):
    #         print("[WARN] MTL still contains absolute-looking paths; fallback expected to textures/.")
    #     if fixed != mtl_path:
    #         try:
    #             fixed.unlink()
    #         except Exception:
    #             pass
    # except Exception as e:
    #     print(f"[WARN] failed to overwrite original MTL with fixed: {e}")
        
    return obj_path

from pathlib import Path

def import_and_sanitize_glb(glb_path):
    """导入一个 glb，合并为单一 MESH，清父子并烘焙变换，最后居中到原点"""
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not meshes:
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True); o.parent = None
    bpy.context.view_layer.objects.active = meshes[0]
    # bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.convert(target='MESH')
    recenter_world(obj)
    return obj

def import_and_sanitize_objdir(obj_dir):
    """导入一个 OBJ 输出目录（包含 model.obj），合并为单一 MESH 并居中"""
    import bpy
    from pathlib import Path

    obj_path = Path(obj_dir) / "model.obj"
    if not obj_path.exists():
        print(f"[WARN] OBJ missing: {obj_path}")
        return None

    # 记录导入前的对象集合，用于稳健地拿到“新导入的对象”
    before = set(bpy.data.objects)

    # —— 关键改动：导入时显式指定轴向，匹配我们导出时的 `Forward=Y, Up=Z`
    try:
        if hasattr(bpy.ops, "wm") and hasattr(bpy.ops.wm, "obj_import"):
            # Blender 4.x 新导入器
            bpy.ops.wm.obj_import(
                filepath=str(obj_path),
                forward_axis='Y',
                up_axis='Z',
            )
    except Exception as e:
        print(f"[ERROR] OBJ import failed: {e}")
        return None

    # 稳健拿到刚导入的对象
    after = set(bpy.data.objects)
    imported = [o for o in (after - before)]
    meshes = [o for o in imported if o.type == 'MESH']
    if not meshes:
        print("[WARN] No mesh objects found after import.")
        return None

    # 统一到单一网格对象
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
        o.parent = None
    bpy.context.view_layer.objects.active = meshes[0]

    # 取消实例化/修改器影响，并把当前变换烘焙到几何
    # bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.convert(target='MESH')

    # 回中心（你已有的函数）
    recenter_world(obj)
    return obj


def import_and_sanitize_item(item):
    """item: {'kind':'glb'|'obj', 'path':str}"""
    if item['kind'] == 'glb':
        return import_and_sanitize_glb(item['path'])
    else:
        return import_and_sanitize_objdir(item['path'])

def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--data_root", required=True, help="where obj stored, for oxl:/home/zby/Simulation-pipeline/Data/datasets/ObjaverseXL_github")
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--grid_n", type=int, default=None, help="网格宽度 N（如 3 表示 3x3）")
    ap.add_argument("--spacing_factor", type=float, default=1.15, help="单元间距倍数，>1 增大空隙")
    ap.add_argument("--snapshot", default="snapshot.png")
    ap.add_argument("--normalize_unit", action="store_true", help="导出前把每个模型最大边归一到 target_size")
    ap.add_argument("--target_size", type=float, default=1.0, help="normalize_unit 时的目标最大边尺寸")
    ap.add_argument("--prefer_gpu", action="store_true", help="尽力启用 GPU 渲染")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing GLBs and snapshot.png")
    ap.add_argument("--clean_out_dir", action="store_true", help="Remove existing exported GLBs in out_dir before running")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N shapes (debug)")
    ap.add_argument("--topdown", action="store_true", help="Use top-down orthographic view for snapshot")
    ap.add_argument("--require_hollow_for", nargs="*", default=None,
                help="哪些类需要做空心+开口筛查（空则用脚本内置默认）")
    ap.add_argument("--export_format", choices=["glb", "obj"], help="Export format for aligned models")

    # align_export_snapshot_blender.py → main() parser
    ap.add_argument("--oxl_layout", action="store_true",
                    help="Use OXL layout: <data_root>/<oxl_out_category>/<label>/<sha>/raw/*")
    ap.add_argument("--oxl_out_category", type=str, default="",
                    help="Base folder name under data_root for OXL downloads (default: oxl)")
    ap.add_argument("--oxl_filter_by_label", action="store_true", default=True,
                help="When --oxl_layout, only process items whose 'label' list contains --category")
    
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fmt = (getattr(args, "export_format", "obj") ).lower()
    report_fp = out_dir/f"align_report_{fmt}.tsv"
    if args.overwrite and report_fp.exists(): report_fp.unlink()
    report_lines = ["shape_key\tstatus\treason\tsrc\n"]
    # if args.clean_out_dir:
    #     for p in out_dir.glob("*.glb"):
    #         try: p.unlink()
    #         except Exception: pass
    #     try:
    #         (out_dir / args.snapshot).unlink(missing_ok=True)
    #     except Exception:
    #         pass
    
    if args.clean_out_dir:
        fmt = (args.export_format or "").lower()
        if fmt == "glb":
            for p in out_dir.glob("*.glb"):
                try: p.unlink()
                except Exception: pass
        elif fmt == "obj":
            for p in out_dir.glob("*.obj"):
                try: p.unlink()
                except Exception: pass
            for p in out_dir.glob("*.mtl"):
                try: p.unlink()
                except Exception: pass
            # OBJ 的 textures 目录按需清理（谨慎，如果同目录还有别的 textures 就不要删）
            # import shutil; shutil.rmtree(out_dir/"textures", ignore_errors=True)

        # 只删“本次格式”的快照
        from pathlib import Path as _P
        stem, suf = _P(args.snapshot).stem, _P(args.snapshot).suffix
        tagged = f"{stem}_{fmt}{suf}" if fmt else args.snapshot
        try:
            (out_dir / tagged).unlink(missing_ok=True)
        except Exception:
            pass
    if isinstance(args.require_hollow_for, list) and args.require_hollow_for:
        req_set = set(s.lower() for s in args.require_hollow_for)
    elif isinstance(args.require_hollow_for, str) and args.require_hollow_for.strip():
        req_set = set(s.strip().lower() for s in args.require_hollow_for.split(","))
    else:
        req_set = REQUIRE_HOLLOW_DEFAULT
    do_hollow_check = args.category.lower() in req_set
    # 读旋转
    def load_rotations_json(json_path: Path) -> dict[str, dict]:
        """
        统一加载新/旧两种结构：
        - 旧：{"export_info": ..., "shape_rotations": {...}}
        - 新：{"<sha>": {"rotation_matrix":..., "label":[...]}, ...}
        返回：{key -> info}（key 保持原样；info 至少含 rotation_matrix 或 R；可含 label）
        """
        import json
        J = json.loads(Path(json_path).read_text(encoding="utf-8"))  # 顶层
        if isinstance(J, dict) and "shape_rotations" in J and isinstance(J["shape_rotations"], dict):
            return J["shape_rotations"]
        # 新结构：顶层就是若干 sha→info；过滤掉 meta 项（如 "export_info"）
        R = {}
        for k, v in (J.items() if isinstance(J, dict) else []):
            if not isinstance(v, dict):
                continue
            # 识别“像旋转项”的条目
            if ("rotation_matrix" in v) or ("R" in v) or ("label" in v) or ("is_identity" in v):
                R[k] = v
        return R

    # J = json.loads(Path(args.json).read_text(encoding="utf-8")) #json items
    # rotations = J.get("shape_rotations", J)  # 兼容两种顶层
    rotations = load_rotations_json(Path(args.json))

    exported = []  # (glb_path, post_dim)
    # report_lines = ["shape_key\tstatus\tinfo"]
    items = list(rotations.items())
    if args.limit: items = items[:args.limit]
    valid_cnt = 0; invalid_cnt = 0; imported_cnt = 0
    # ---------- Pass 1：单体对齐 & 导出 ----------
    for shape_key, info in rotations.items():
        clear_scene_hard()              # 每轮一开始清场
        try:
            try:
                R = np.array(info.get("rotation_matrix", info.get("R", info)), dtype=float).reshape(3,3)
            except Exception as e:
                report_lines.append(f"{shape_key}\tbad_rotation\t{e}")
                continue

            # 2) OXL 模式：可选按 label 过滤
            if args.oxl_layout and args.oxl_filter_by_label and args.category:
                labs = info.get("label", []) or []
                if isinstance(labs, str):
                    labs = [labs]
                if args.category not in labs:
                    continue    
            # 3) 解析实例目录 & 导出名
            data_root = Path(args.data_root)
            if args.oxl_layout:
                parts = str(shape_key).split("/")  # 允许 "label/sha" 或 纯 "sha"
                if len(parts) == 2:
                    lab, sha = parts[0], parts[1]
                else:
                    # 纯 sha：优先 info['label'][0]，否则用当前 --category
                    l0 = info.get("label", [None])[0] if isinstance(info.get("label"), list) else info.get("label")
                    lab = l0 or (args.category or "")
                    sha = parts[0]
                inst_dir = data_root / args.oxl_out_category / lab / sha
                export_stem = f"{lab}_{sha}"  
            else:  
                try:
                    subA, subB = shape_key.split("/")
                except Exception as e:
                    report_lines.append(f"{shape_key}\tbad_shape_key\t{e}")
                    continue
                inst_dir = data_root / args.category / subA / subB
                export_stem = f"{args.category}_{subA}_{subB}"        
            # inst_dir = data_root / args.category / subA / subB
            mesh_path = find_mesh_file(inst_dir)
            if not mesh_path:
                if args.verbose: print(f"[SKIP] {shape_key}: no_mesh {inst_dir}")
                report_lines.append(f"{shape_key}\tinvalid\tno_mesh\t{inst_dir}\n"); invalid_cnt += 1
                continue
            if "--verbose" in argv:
                print(f"[DBG] shape={shape_key}")
                print(f"[DBG] inst_dir={inst_dir} exists={inst_dir.exists()}")
           
            # src = None
            # for ext in PRIORITY_EXTS:
            #     cands = sorted(inst_dir.rglob(f"*{ext}"), key=lambda p:(len(p.parts),len(str(p))))
            #     if cands: src = cands[0]; break
            # if not src:
            #     if args.verbose: print(f"[SKIP] {shape_key}: no_mesh {inst_dir}")
            #     report_lines.append(f"{shape_key}\tinvalid\tno_mesh\t{inst_dir}\n"); invalid_cnt += 1
            #     continue
            # 4) 找 mesh（raw 下优先）
            src = Path(mesh_path) if mesh_path else None
            
            if not src:
                # 兜底：全目录按优先扩展找
                src = None
                for ext in PRIORITY_EXTS:
                    cands = sorted(inst_dir.rglob(f"*{ext}"), key=lambda p:(len(p.parts),len(str(p))))
                    if cands:
                        src = cands[0]; break
            if not src:
                if args.verbose: print(f"[SKIP] {shape_key}: no_mesh {inst_dir}")
                report_lines.append(f"{shape_key}\tinvalid\tno_mesh\t{inst_dir}\n"); invalid_cnt += 1
                continue
            # clear_scene()
            src_base = Path(src).stem
            if src_base.endswith("_fixed"):
                src_base = src_base[:-6]  # 去掉 "_fixed"
            try:
                objs = import_mesh(str(src))
            
                # === 新增：导入后统一重连（OBJ/FBX/GLB） ===
                # 导入后：先把骨骼/动画/相机/灯光处理掉，并把形变烘焙到网格
                # try:
                #     bake_skins_and_strip_non_meshes()
                # except Exception as e:
                #     if args.verbose:
                #         print(f"[SANITIZE] strip_rig failed: {e}")
                try:
                    raw_dir = inst_dir / "raw"
                    fbm_dir = None
                    if Path(src).suffix.lower() == ".fbx":
                        fbm_dir = Path(src).with_suffix(".fbm")
                    relink_images(raw_dir, fbm_dir)
                    # 保存 → 相对化（避免 "Cannot set relative paths..."）
                    tmp_blend = raw_dir / "_import_tmp.blend"
                    bpy.ops.wm.save_mainfile(filepath=str(tmp_blend))
                    with contextlib.suppress(Exception):
                        bpy.ops.file.make_paths_relative()
                        bpy.ops.file.pack_all()
                except Exception as _e:
                    pass
                
                main_obj = join_imported_meshes(objs)
                check_materials(main_obj)
                if main_obj is None:
                    report_lines.append(f"{shape_key}\tno_mesh_object\t-"); invalid_cnt += 1
                    continue
                # realize_and_make_single_user() # multiple call of make_single_user() will cause error
                # 旋转（绕原点）
            
            except Exception as e:
                report_lines.append(f"{shape_key}\timport_fail\t{e}\n"); invalid_cnt += 1
                continue

            # main_obj = next((o for o in objs if o.type=='MESH'), None)
            main_obj = bpy.context.view_layer.objects.active

            if main_obj is None:
                report_lines.append(f"{shape_key}\tno_mesh_object\t-")
                continue
            
            imported_cnt += 1
            
            # # 旋转（绕原点）
            R_orth = orthonormalize_R_if_needed(R) # for unorthogonal matrices if needed
            main_obj.matrix_world = to_M4_from_R(R) @ main_obj.matrix_world # R
             # 旋转对齐
            # align_and_rotate(main_obj, R)
            # recenter_world(main_obj)
            # # 保证导出 GLB 尺度一致）
            # if args.normalize_unit:
            #     bpy.context.view_layer.objects.active = main_obj
            #     bpy.ops.object.make_single_user(type='ALL', object=True, obdata=True)
            #     bpy.ops.object.convert(target='MESH')
            #     # —— 防止 0 / NaN 尺寸
            #     dims = main_obj.dimensions
            #     max_dim = max(dims.x, dims.y, dims.z)
            #     if not (max_dim and max_dim > 1e-8):
            #         print(f"[SKIP] zero/invalid size after import for {shape_key}")
            #         report_lines.append(f"{shape_key}\tinvalid_size\t-")
            #         invalid_cnt += 1
            #         continue
            #     normalize_object(main_obj, target_size=args.target_size)
            if args.normalize_unit:
                ok, info = normalize_unit_pipeline(main_obj, target_size=args.target_size)
                if not ok:
                    print(f"[SKIP] invalid size after import for {shape_key}")
                    report_lines.append(f"{shape_key}\tinvalid_size\t-")
                    invalid_cnt += 1
                    continue
            # 导出 GLB（只包含旋转/尺度，不包含拼图平移）
            has_glb = any(out_dir.glob("*.glb"))
            has_obj = any(p.is_dir() and p.name.endswith("_obj") for p in out_dir.iterdir())
            # 法线 & single user
            # make_mesh_single_user(main_obj) # called multiple times could break the mesh
            fix_normals(main_obj) 

            # —— 导出：按格式分支，且 overwrite 仅覆盖同类输出
            if args.export_format == "glb":
                out_glb = out_dir / f"{export_stem}.glb"
                if args.overwrite and out_glb.exists():
                    try: out_glb.unlink()
                    except Exception as e: print(f"[WARN] rm {out_glb}: {e}")

                export_only_selected_glb(main_obj, out_glb)
                # 仅当导出成功才计入
                if out_glb.exists() and out_glb.stat().st_size > 0:
                    exported.append({'kind':'glb', 'path': str(out_glb), 'dim': float(max(main_obj.dimensions[:]))})
                    report_lines.append(f"{shape_key}\tvalid\tok\t{src}\n")
                    valid_cnt += 1
                else:
                    report_lines.append(f"{shape_key}\texport_fail\tno_glb\t{out_glb}\n")
                    invalid_cnt += 1

            elif args.export_format == "obj":
                out_obj_dir = out_dir / f"{export_stem}_obj"
                # 只在 overwrite 时移除该实例的 obj 目录；不影响别的实例
                if args.overwrite and out_obj_dir.exists():
                    try: shutil.rmtree(out_obj_dir)
                    except Exception as e: print(f"[WARN] rm -r {out_obj_dir}: {e}")

                try:
                    export_selected_as_obj_with_tfextures(main_obj, out_obj_dir, name="model") #name = src_base
                    model_obj = out_obj_dir / "model.obj"
                    if model_obj.exists():
                        exported.append({'kind':'obj', 'path': str(out_obj_dir), 'dim': float(max(main_obj.dimensions[:]))})
                        report_lines.append(f"{shape_key}\tobj_ok\t{model_obj}\n")
                        valid_cnt += 1
                    else:
                        report_lines.append(f"{shape_key}\tobj_export_fail\tno_obj\t{out_obj_dir}\n")
                        invalid_cnt += 1
                except Exception as e:
                    report_lines.append(f"{shape_key}\tobj_export_fail\t{e}\n")
                    invalid_cnt += 1

        except Exception as e:
            # 任何未捕获异常都在这里兜底记录，避免整批中断
            report_lines.append(f"{shape_key}\tinvalid\texception\t{repr(e)}\n")
            invalid_cnt += 1

        finally:
            # —— 每轮结束：无论成功失败都清场，避免把本轮对象带到下一轮或误导出
            clear_scene_hard()

    report_fp.write_text("".join(report_lines), encoding="utf-8")
    print(f"[SUMMARY] imported={imported_cnt}  valid={valid_cnt}  invalid={invalid_cnt}")
    print(f"[OK] report: {report_fp}")

    if not exported:
        print("No shapes exported; stop.")
        return

    # ---------- Pass 2：拼图 & 快照 ----------

    # —— 快照拼图：
    # ---------- Pass 2：拼图 & 快照 ----------
    clear_scene_hard()
    set_cycles(prefer_gpu=args.prefer_gpu)
    add_lights_and_world()

    # 清理坏条目（防御性）
    exported = [it for it in exported if it.get('path') and Path(it['path']).exists()]
    if not exported:
        print("No shapes exported; stop.")
        report_fp.write_text("".join(report_lines), encoding="utf-8")
        print(f"[SUMMARY] imported={imported_cnt}  valid={valid_cnt}  invalid={invalid_cnt}")
        print(f"[OK] report: {report_fp}")
        return

    N_ = args.grid_n or int(math.ceil(math.sqrt(len(exported))))
    sizes = [it['dim'] for it in exported if it.get('dim') and it['dim'] > 0]
    base = float(np.median(sizes)) if sizes else 1.0
    cell = base * args.spacing_factor

    offsets = center_symmetric_grid_offsets(len(exported), N_, cell)
    all_min = Vector((1e9, 1e9, 1e9))
    all_max = Vector((-1e9,-1e9,-1e9))
    placed = 0

    for item, ofs in zip(exported, offsets):
        obj = import_and_sanitize_item(item)
        if not obj:
            continue
        dim = obj.dimensions
        if max(dim.x, dim.y, dim.z) <= 1e-8:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue
        obj.location = Vector(ofs)
        for v in obj.bound_box:
            w = obj.matrix_world @ Vector(v)
            all_min.x = min(all_min.x, w.x); all_max.x = max(all_max.x, w.x)
            all_min.y = min(all_min.y, w.y); all_max.y = max(all_max.y, w.y)
            all_min.z = min(all_min.z, w.z); all_max.z = max(all_max.z, w.z)
        placed += 1

    fmt = (args.export_format or "unknown").lower()
    snap_path = out_dir / f"{_P(args.snapshot).stem}_{fmt}{_P(args.snapshot).suffix}"
    if args.overwrite and snap_path.exists():
        try: snap_path.unlink()
        except Exception: pass

    if placed == 0:
        print("[SNAP] no objects placed; skip rendering")
    else:
        span_xy = max((all_max - all_min).x, (all_max - all_min).y)
        axes_size = max(1.0, span_xy * 5.0)
        # add_world_axes(size=axes_size)
        place_camera_fit(all_min, all_max, snap_path, N=N_, topdown=args.topdown, verbose=args.verbose)
        bpy.ops.render.render(write_still=True)
        print(f"[OK] snapshot saved: {snap_path}")

    print(f"[OK] exported: {len(exported)} → {out_dir}")


if __name__ == "__main__":
    main()