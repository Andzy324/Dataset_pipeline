#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, shutil, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = Path(__file__).parent

def sh(cmd):
    print("[run]", " ".join(map(str, cmd)))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")

def guess_class_and_stem(name: str):
    """
    从 stem（形如 mug_965_4831669）解析：
    - class = 'mug'
    - stem  = 原始完整名
    """
    if "_" in name:
        cls = name.split("_", 1)[0]
        return cls, name
    return None, name

def discover_jobs(pc_root: Path, mesh_root: Path, classes=None):
    """
    在 pc_root 下找每个形状文件夹（含 points.ply / labels.npy），
    匹配 mesh_root/<class>/<stem>.glb
    返回：{class: [(stem, pc_path, lab_path, glb_path), ...]}
    """
    jobs = {}
    for d in sorted(pc_root.iterdir()):
        if not d.is_dir():
            continue
        stem = d.name
        pc   = d / "points.ply"
        lab  = d / "labels.npy"
        if not pc.is_file() or not lab.is_file():
            print(f"[skip] missing pc/labels in {d}")
            continue
        cls, _ = guess_class_and_stem(stem)
        if cls is None:
            print(f"[skip] cannot parse class from {stem}")
            continue
        if classes and cls not in classes:
            continue
        glb = mesh_root / cls / f"{stem}.glb"
        if not glb.is_file():
            print(f"[skip] missing GLB: {glb}")
            continue
        jobs.setdefault(cls, []).append((stem, pc, lab, glb))
    return jobs

def preset_to_rx_deg(preset: str) -> float:
    """
    将友好的轴系预设映射到 ct_pc_rot.py 的 --rx_deg（绕X旋转，度）
    - 'yup_to_zup'  : +90（你的点云Z-up，GLB看似-90°；恢复性+90）
    - 'zup_to_yup'  : -90
    - 'none'        : 0
    """
    p = (preset or "none").lower()
    if p == "yup_to_zup": return 90.0
    if p == "zup_to_yup": return -90.0
    return 0.0

def main():
    ap = argparse.ArgumentParser("Batch color transfer (multi-class) for ct_pc_rot.py")
    # 数据根
    ap.add_argument("--pc_root",  required=True, type=Path,
                    help="如 /home/.../Data/copca_mug/beta_0.7 （其下是 <class>_<id>/points.ply & labels.npy）")
    ap.add_argument("--mesh_root", required=True, type=Path,
                    help="如 /home/.../Data/selected_aligned_glb （其下是 <class>/<class>_<id>.glb）")
    ap.add_argument("--out_root",  required=True, type=Path,
                    help="输出根：<out_root>/<class>/{glb|obj}")

    # 类别过滤（可选）
    ap.add_argument("--classes", type=str, default=None,
                    help="仅处理这些类，逗号分隔，如 mug,bowl；缺省自动发现全部")
    ap.add_argument("--workers", type=int, default=1,
                    help="并发上色任务数（I/O+CPU混合；2~4 视机器而定）")
    ap.add_argument("--overwrite", action="store_true")

    # —— 映射到 ct_pc_rot.py 的参数（保持一致）——
    ap.add_argument("--rx_preset", default="yup_to_zup",
                    help="轴系预设：yup_to_zup / zup_to_yup / none（映射到 --rx_deg）")
    ap.add_argument("--pivot", choices=['pc_centroid','mesh_centroid','origin'],
                    default="pc_centroid")
    ap.add_argument("--icp", action="store_true", help="执行ICP微调")
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--max_dist", type=float, default=0.04)
    ap.add_argument("--neighbor_mode", choices=['knn','radius'], default='radius')
    ap.add_argument("--radius", type=float, default=0.006)
    ap.add_argument("--trim_frac", type=float, default=0.5)
    ap.add_argument("--min_inlier_frac", type=float, default=0.7)
    ap.add_argument("--normal_consistency_deg", type=float, default=None)

    ap.add_argument("--paint_mode", choices=['rgb_weighted','label_vertex','label_facecut'],
                    default='label_facecut')
    ap.add_argument("--only_labels", type=int, nargs="*", default=None,
                    help="仅从这些 labels 的点上色，如把手=1、杯身=0 等")
    ap.add_argument("--base_rgb", type=str, default="200,200,200")
    ap.add_argument("--vcol_threshold", type=float, default=0.02)

    ap.add_argument("--make_obj", action="store_true",
                    help="要求同时导出 OBJ")
    ap.add_argument("--obj_mode", choices=['baked','vcol'], default='vcol',
                    help="OBJ 导出方式：vcol=OBJ顶点色；baked=贴图OBJ")

    # 渲染（可选，对接你提供的 render_orbit_full.py；默认渲 GLB）
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render_input", choices=['glb','obj'], default='glb')
    ap.add_argument("--render_script", type=Path, default=HERE / "render_orbit_full.py")
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--num_cams", type=int, default=80)
    ap.add_argument("--elev_deg", type=float, default=20.0)
    ap.add_argument("--batch_chunk", type=int, default=32)
    ap.add_argument("--bin_size", type=int, default=0)
    ap.add_argument("--max_faces_per_bin", type=int, default=20000)
    ap.add_argument("--bg_color", type=float, nargs=3, default=[1,1,1])

    args = ap.parse_args()

    pc_root   = args.pc_root.resolve()
    mesh_root = args.mesh_root.resolve()
    out_root  = args.out_root.resolve()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    jobs_by_cls = discover_jobs(pc_root, mesh_root, classes)
    if not jobs_by_cls:
        print(f"[exit] no jobs under {pc_root}")
        return

    # 预建输出结构
    for cls in jobs_by_cls:
        (out_root/cls/"glb").mkdir(parents=True, exist_ok=True)
        (out_root/cls/"obj").mkdir(parents=True, exist_ok=True)
        (out_root/"_work"/cls).mkdir(parents=True, exist_ok=True)

    def one_job(cls, stem, pc, lab, glb):
        cls_root = out_root/cls
        workdir  = out_root/"_work"/cls/stem
        if workdir.exists() and args.overwrite:
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        out_glb = cls_root/"glb"/f"{stem}.glb"
        if out_glb.exists() and not args.overwrite:
            print(f"[skip] exists: {out_glb}")
            return {"class": cls, "stem": stem, "glb": str(out_glb), "obj": None}

        # —— 调用你的 ct_pc_rot.py ——（严格按其现有参数命名）
        cmd = [
            sys.executable, str(HERE/"ct_pc_rot.py"),
            "--pc", str(pc),
            "--glb", str(glb),
            "--workdir", str(workdir),
            "--rx_deg", str(preset_to_rx_deg(args.rx_preset)),
            "--pivot", args.pivot,
            "--knn", str(args.knn),
            "--max_dist", str(args.max_dist),
            "--neighbor_mode", args.neighbor_mode,
            "--radius", str(args.radius),
            "--trim_frac", str(args.trim_frac),
            "--min_inlier_frac", str(args.min_inlier_frac),
            "--base_rgb", args.base_rgb,
            "--vcol_threshold", str(args.vcol_threshold),
            "--paint_mode", args.paint_mode,
            "--labels", str(lab),
            "--tex_res", "2048",
        ]
        # ICP 微调
        if args.icp:
            cmd.append("--icp")
        # 仅着色这些 labels
        if args.only_labels:
            cmd += ["--only_labels"] + list(map(str, args.only_labels))
        # OBJ 请求
        if args.make_obj:
            cmd += ["--make_obj", "--obj_mode", args.obj_mode]

        sh(cmd)

        # —— 收集产物到目标结构 —— 
        # 你的脚本把 GLB 写在 workdir/colored_aligned.glb；复制到 <out_root>/<class>/glb/<stem>.glb
        src_glb = workdir/"colored_aligned.glb"
        if src_glb.is_file():
            shutil.copy2(src_glb, out_glb)
        else:
            # 有些版本也可能直接在 workdir 里导出同名；这里加个兜底
            cand = list(workdir.glob("*.glb"))
            if cand:
                shutil.copy2(cand[0], out_glb)
        out_obj = None

        if args.make_obj:
            # 你的脚本（vcol 模式）将 OBJ 写到 workdir/_obj/model_vcol.obj
            # baked 模式则是 workdir/_obj/model.obj (+ textures/)
            obj_dir = workdir/"_obj"
            if args.obj_mode == "vcol":
                src_obj = obj_dir/"model_vcol.obj"
                if src_obj.is_file():
                    out_obj = cls_root/"obj"/f"{stem}_vcol.obj"
                    shutil.copy2(src_obj, out_obj)
            else:
                # baked：把整个 _obj 目录拷到 <class>/obj/<stem>/...
                if obj_dir.is_dir():
                    dst_dir = cls_root/"obj"/stem
                    if dst_dir.exists():
                        shutil.rmtree(dst_dir)
                    shutil.copytree(obj_dir, dst_dir)
                    out_obj = dst_dir/"model.obj"

        return {"class": cls, "stem": stem, "glb": str(out_glb), "obj": (str(out_obj) if out_obj else None)}

    results = []
    if args.workers <= 1:
        for cls, lst in jobs_by_cls.items():
            for stem, pc, lab, glb in lst:
                try:
                    results.append(one_job(cls, stem, pc, lab, glb))
                except Exception as e:
                    print(f"[ERR] {cls}/{stem} :: {e}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for cls, lst in jobs_by_cls.items():
                for stem, pc, lab, glb in lst:
                    futs.append(ex.submit(one_job, cls, stem, pc, lab, glb))
            for fu in as_completed(futs):
                try:
                    results.append(fu.result())
                except Exception as e:
                    print("[ERR]", e)

    # 清单
    (out_root).mkdir(parents=True, exist_ok=True)
    with open(out_root/"manifest_all.json", "w") as f:
        json.dump(results, f, indent=2)
    by_cls = {}
    for r in results:
        by_cls.setdefault(r["class"], []).append(r)
    for cls, lst in by_cls.items():
        with open(out_root/f"manifest_{cls}.json", "w") as f:
            json.dump(lst, f, indent=2)
    print("[done] manifests:", out_root/"manifest_all.json", " & per-class manifests")

    # —— 可选批量渲染（建议渲 GLB）——
    if args.render:
        for cls in by_cls:
            in_dir  = out_root/cls/("obj" if args.render_input=="obj" else "glb")
            out_dir = out_root/cls/f"render_{args.render_input}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable, str(args.render_script),
                "--obj", str(in_dir),
                "--out", str(out_dir),
                "--input_format", args.render_input,     # glb 或 obj
                "--image_size", str(args.image_size),
                "--num_cams", str(args.num_cams),
                "--elev_deg", str(args.elev_deg),
                "--batch_chunk", str(args.batch_chunk),  # 显存阀门
                "--bin_size", str(args.bin_size),
                "--max_faces_per_bin", str(args.max_faces_per_bin),
                "--bg_color", *map(str, args.bg_color),
                "--no_cull_backfaces",
                "--overwrite",
            ]
            sh(cmd)
            print(f"[done] render {cls} -> {out_dir}")

if __name__ == "__main__":
    main()
