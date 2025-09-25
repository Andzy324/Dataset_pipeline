#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end mesh workflow runner for your three steps:
1) download → 2) align & export GLBs in Blender → 3) render orbit RGBD & video.

It orchestrates your existing scripts:
- build_metadata.py
- align_export_snapshot_blender.py (run via Blender -b -P ... -- ...)
- render_orbit_rgbd.py

Control which steps run with `--steps` (any order subset): `download align render`.

You can also run a **per-category flow** driven by `--rot_dir`, which for each category executes: `download → align → render` before moving to the next one. Enable with `--per_category_flow`. This keeps the sub-scripts unchanged while the pipeline iterates categories based on `--rot_dir/<category>/<rot_filename>`.
Examples:
- Only download: `--steps download`
- Only align: `--steps align --categories mug`
- Only render (from existing GLBs): `--steps render --render_scheme hier`
- Align+Render interleaved per category: `--steps align render --render_after_each_category`

Folder layout (default under --root):
  root/
    categories../                # raw assets & metadata.csv produced by build_metadata.py
    aligned_glb/<cat>/        # GLBs exported by Blender + snapshot.png
    renders/<cat>/<...>/      # per-shape renders (configurable nesting)
#     logs/                     # logs & reports
#     manifests/                # run manifests

Usage (example):
  python mesh_workflow_pipeline.py \
    --root /data/mesh_pipeline \
    --png_root /Simulation-pipeline/Data/G-objaverse \
    --maps_json /path/to/all_categories_id_to_urls.json \
    --rot_json  /path/to/shape_rotations.json \
    --categories mug bottle teacup \
    --blender /snap/blender/current/blender \
    --normalize_unit --target_size 1.0 \
    --num_cams 80 --elev_deg 20 --image_size 512 --fov_deg 60 \
    --make_video --save_metric_depth

Notes:
- Pass your actual Blender binary path via --blender.
- This script is idempotent: it skips work if outputs already exist unless --overwrite.
- You can choose render storage style with --render_scheme {flat,hier}.
"""

from __future__ import annotations
import argparse, json, os, re, shutil, subprocess, sys, time, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Optional, Union, Dict, Any

# ---------- Utilities ----------
# --- NEW: YAML & timing helpers ---
try:
    import yaml  # pip install pyyaml
except Exception as _e:
    yaml = None

from time import perf_counter
OXL_OUT_CAT = "aligned_xl"

def sh(cmd: List[str], dry: bool=False, cwd: Optional[Path]=None, quiet: bool=False) -> int:
    """
    Run a shell command with optional quiet mode.
    - quiet=False: 打印 `$ cmd`，并透传子进程输出到终端（默认）
    - quiet=True : 不打印命令，且吞掉子进程输出（只在失败时打印一行错误）
    返回：子进程返回码
    """
    if not quiet:
        print("$", " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0

    start = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    if quiet:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
        )

    dur = time.perf_counter() - start
    if proc.returncode != 0:
        print(f"[ERROR] Command failed (rc={proc.returncode}, {dur:.2f}s): {' '.join(str(c) for c in cmd)}")
    else:
        if not quiet:
            print(f"[done] rc=0  time={dur:.2f}s", flush=True)
    return proc.returncode



def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Helpers for OXL branch (download → extract → pick mesh) ----------
# --- NEW: split a label_json (all categories) into per-label small JSONs ---
from pathlib import Path
def split_label_json_per_category(label_json: Path, out_dir: Path) -> dict[str, Path]:
    """
    读取“总领式”label_json（新/旧结构均可），按 label 拆分成每类一份：
      输出统一为 {"shape_rotations": {<sha>: {"rotation_matrix": 3x3}}}
    返回: {label: json_path}
    """
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    J = json.loads(Path(label_json).read_text(encoding="utf-8"))

    # 统一拿到 rotations dict：老格式在 "shape_rotations"，新格式顶层就是 sha→info
    if isinstance(J, dict) and isinstance(J.get("shape_rotations"), dict):
        R = J["shape_rotations"]
    else:
        # 过滤掉 export_info 等非对象项
        R = {k: v for k, v in (J.items() if isinstance(J, dict) else []) if isinstance(v, dict)}

    buckets: dict[str, dict] = {}  # label -> {sha: {"rotation_matrix": ...}}
    for sha, info in R.items():
        # rotation 矩阵容差取法：rotation_matrix / R / 直接 3x3
        rot = info.get("rotation_matrix", info.get("R", info))
        labels = info.get("label", []) or []
        if isinstance(labels, str):
            labels = [labels]
        # 没有 label 就不分类；若需要可落到 "unknown"
        for lab in labels:
            if not lab:
                continue
            buckets.setdefault(lab, {})[sha] = {"rotation_matrix": rot}

    # 写出 per-label 小 JSON
    out_map: dict[str, Path] = {}
    for lab, payload in buckets.items():
        obj = {"shape_rotations": payload}
        dst = out_dir / f"rotations_{lab}.json"
        dst.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        out_map[lab] = dst
    return out_map
# --- JOIN labels from --oxl_labels and --categories; accept comma-separated entries too. ---
def _norm_label_list(xs):
    """
    接受 None / 字符串 / 列表；支持逗号或空格写法；去重保序。
    例: "drill,hammer" -> ["drill","hammer"]; ["drill","hammer"] -> 原样。
    """
    out = []
    if xs is None:
        return out
    if isinstance(xs, (list, tuple)):
        seq = xs
    else:
        seq = [xs]
    for x in seq:
        # 先按逗号拆，再 strip
        parts = [p.strip() for p in str(x).split(",") if p and p.strip()]
        # 如果用户用空格分了（来自 nargs="*"），这里保持原状；逗号写法统一被拆开
        out.extend(parts if parts else [str(x).strip()])
    # 去重保序
    seen = set(); dedup = []
    for v in out:
        if v and v not in seen:
            seen.add(v); dedup.append(v)
    return dedup

def _is_archive_name(name: str) -> bool:
    low = name.lower()
    return low.endswith(('.zip', '.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz', '.tar.bz2', '.tbz'))

def _safe_extract_archive(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    low = archive_path.name.lower()
    if low.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(out_dir)
        return
    if any(low.endswith(ext) for ext in ['.tar', '.tar.gz', '.tgz', '.tar.xz', '.txz', '.tar.bz2', '.tbz']):
        import tarfile
        mode = 'r'
        if low.endswith(('.tar.gz', '.tgz')): mode = 'r:gz'
        elif low.endswith(('.tar.xz', '.txz')): mode = 'r:xz'
        elif low.endswith(('.tar.bz2', '.tbz')): mode = 'r:bz2'
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(out_dir)
        return
    # fallback to 7z
    try:
        subprocess.run(['7z', 'x', str(archive_path), f'-o{str(out_dir)}', '-y'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"Unsupported archive format and 7z failed: {archive_path} ({e})")

_MESH_EXT_PRIORITY = ['.glb', '.gltf', '.obj', '.fbx', '.stl', '.ply', '.blend']

def _pick_mesh_file(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    for ext in _MESH_EXT_PRIORITY:
        matches = list(root.rglob(f"*{ext}"))
        if matches:
            matches.sort(key=lambda p: (len(p.parts), len(str(p))))
            return matches[0]
    return None


def parse_glb_basename(basename: str) -> Tuple[str, str, str]:
    """Parse names like 'mug_978_4895636.glb' → (cat, subA, subB).
    Robust to extra dots by using stem then splitting by underscores from the right.
    """
    stem = Path(basename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"GLB name doesn't look like <cat>_<A>_<B>: {basename}")
    cat = parts[0]
    subB = parts[-1]
    subA = parts[-2]
    return cat, subA, subB


def render_out_dir_for(glb_path: Path, renders_root: Path, scheme: str) -> Path:
    cat, subA, subB = parse_glb_basename(glb_path.name)
    if scheme == "flat":
        # e.g., renders/mug/mug_978_4895636/
        return renders_root / cat / glb_path.stem
    elif scheme == "hier":
        # e.g., renders/mug/978/4895636/
        return renders_root / cat / subA / subB
    else:
        raise ValueError("render_scheme must be 'flat' or 'hier'")

def parse_obj_dirname(dir_name: str) -> Tuple[str, str, str]:
    """
    解析 '<cat>_<A>_<B>_obj' → (cat, A, B)
    """
    name = dir_name[:-4] if dir_name.endswith("_obj") else dir_name
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"OBJ dir name doesn't look like <cat>_<A>_<B>_obj: {dir_name}")
    cat = parts[0]
    subB = parts[-1]
    subA = parts[-2]
    return cat, subA, subB

def render_out_dir_for_any(model_path: Path, renders_root: Path, scheme: str,
                           dl_backend: str = "custom",
                           oxl_out_cat: Optional[str] = None) -> Path:
    """
    统一给 .glb / .obj 生成渲染输出目录：
    - dl_backend == "oxl": 解析 (label, sha) → hier: <renders>/<label>/<sha>/；flat: <renders>/<label>_<sha>/
    - 否则：沿用旧逻辑
      · GLB: 由文件名 '<cat>_<A>_<B>.glb'
      · OBJ: 由父目录名 '<cat>_<A>_<B>_obj'
    """
    p = Path(model_path)

    # --- OXL 优先 ---
    if dl_backend == "oxl":
        toks = _infer_label_sha_from_path(p, oxl_out_cat=oxl_out_cat)
        if toks:
            label, sha = toks
            if scheme == "flat":
                return renders_root / label /f"{label}_{sha}"
            elif scheme == "hier":
                return renders_root / label / sha
            else:
                raise ValueError("render_scheme must be 'flat' or 'hier'")

    # --- 旧逻辑：保持不变 ---
    if p.suffix.lower() == ".glb":
        return render_out_dir_for(p, renders_root, scheme)
    elif p.suffix.lower() == ".obj":
        cat, subA, subB = parse_obj_dirname(p.parent.name)
        if scheme == "flat":
            return renders_root / cat / f"{cat}_{subA}_{subB}"
        elif scheme == "hier":
            return renders_root / cat / subA / subB
        else:
            raise ValueError("render_scheme must be 'flat' or 'hier'")
    else:
        raise ValueError(f"Unsupported model type for render out dir: {p}")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)
def _is_sha256(s: str) -> bool:
    return bool(_SHA256_RE.match(s or ""))

def _infer_label_sha_from_path(p: Path, oxl_out_cat: Optional[str] = None) -> Optional[Tuple[str,str]]:
    """
    适配 OXL 命名/路径，返回 (label, sha) 或 None：
      1) <label>_<sha>.glb / .gltf / .obj
      2) 父目录名 <label>_<sha>_obj
      3) 原始布局 .../<oxl_out_category>/<label>/<sha>/raw/...
         （oxl_out_category 可省略，不做强校验）
    """
    p = Path(p)

    # 情况 1：文件名 <label>_<sha>.<ext>
    stem = p.stem
    if "_" in stem:
        lab, sha = stem.rsplit("_", 1)
        if _is_sha256(sha):
            return (lab, sha)

    # 情况 2：父目录 <label>_<sha>_obj
    parent = p if p.is_dir() else p.parent
    name = parent.name
    if name.lower().endswith("_obj"):
        core = name[:-4]
        if "_" in core:
            lab, sha = core.rsplit("_", 1)
            if _is_sha256(sha):
                return (lab, sha)

    # 情况 3：原始路径 .../<out_cat>/<label>/<sha>/raw/...
    parts = list(p.parts)
    try:
        ridx = len(parts) - 1 - parts[::-1].index("raw")
        if ridx >= 2:
            sha = parts[ridx-1]; lab = parts[ridx-2]
            if _is_sha256(sha):
                # 可选：若提供了 oxl_out_category，可进一步核对 parts[ridx-3] 是否匹配
                return (lab, sha)
    except ValueError:
        pass

    return None

# ---------- Pipeline Steps ----------

def _oxl_metadata_csv_path(args) -> Path:
    root = Path(args.download_root).resolve()
    out_cat = str(getattr(args, "oxl_out_category", "oxl"))
    return root / out_cat / "metadata.csv"

def _has_complete_metadata_csv(csv_path: Path, required=("sha256","source_url")) -> bool:
    try:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return False
        try:
            import pandas as _pd
        except ImportError:
            return False
        df = _pd.read_csv(csv_path)
        return all(c in df.columns for c in required) and len(df) > 0
    except Exception:
        return False

def iter_objs(aligned_cat_dir: Path) -> Iterable[Path]:
    """
    遍历对齐导出的 OBJ 目录（形如 '<cat>_<A>_<B>_obj'），返回其中的 model.obj（或首个 .obj）
    """
    for d in sorted([p for p in aligned_cat_dir.iterdir() if p.is_dir() and p.name.endswith("_obj")]):
        # 优先 'model.obj'，否则取目录内首个 .obj
        cand = d / "model.obj"
        if cand.exists():
            yield cand
            continue
        others = sorted(d.glob("*.obj"))
        if others:
            yield others[0]


def resolve_rot_json(args, category: str) -> Path:
    """Pick the rotation JSON for a given category.
    Priority:
      1) --rot_dir/<category>/<rot_filename>
      2) --rot_json fallback
    """
    if getattr(args, "rot_dir", None):
        base = Path(args.rot_dir).resolve()
        fp = base / category / (args.rot_filename or "final_shape_rotations.json")
        if fp.exists():
            return fp.resolve()
        raise FileNotFoundError(f"Rotation JSON not found: {fp}")
    if getattr(args, "rot_json", None):
        fp = Path(args.rot_json).resolve()
        if fp.exists():
            return fp
    raise FileNotFoundError(
        f"No rotation JSON found for category '{category}'. Provide --rot_dir or --rot_json.")

def infer_categories_from_rot_dir(rot_dir: Path, rot_filename: str) -> List[str]:
    """Return categories as immediate subdirectories of rot_dir that contain rot_filename."""
    rot_dir = Path(rot_dir).resolve()
    cats: List[str] = []
    for d in sorted([p for p in rot_dir.iterdir() if p.is_dir()]):
        if (d / rot_filename).exists():
            cats.append(d.name)
    return cats

def _load_json_loose(fp: Path) -> Dict[str, Any]:
    """宽松读取：如果顶层有 'shape_rotations' 就取它；否则尝试把顶层当作映射用。"""
    with open(fp, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "shape_rotations" in data and isinstance(data["shape_rotations"], dict):
        return data["shape_rotations"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"{fp} 不是可识别的 rotations 格式（需为 dict 或 dict.shape_rotations）")

def _is_R_ok(R):
    return isinstance(R, list) and len(R)==3 and all(isinstance(r, list) and len(r)==3 for r in R)

def _merge_rot_maps(files: List[Path], dup_policy: str = "prefer_last") -> Dict[str, Any]:
    """合并多个 rot 映射；重复键按策略处理。"""
    merged: Dict[str, Any] = {}
    for fp in files:
        m = _load_json_loose(fp)
        for sid, info in m.items():
            # 可选：校验 3x3
            if isinstance(info, dict) and "rotation_matrix" in info and not _is_R_ok(info["rotation_matrix"]):
                print(f"[WARN] 非法 3x3 矩阵，跳过 {sid} in {fp.name}")
                continue
            if sid in merged:
                if dup_policy == "error":
                    raise ValueError(f"重复 shape id {sid} 出现在 {fp.name}")
                elif dup_policy == "prefer_first":
                    continue
                # 默认 prefer_last：覆盖
            merged[sid] = info
    return merged

def resolve_or_merge_rot_json(
    args,
    category: str,
    merge_glob: Optional[str] = None,
    dup_policy: Optional[str] = None,
    persist: Optional[bool] = None,
    force_remerge: bool = False,
) -> Path:
    """
    返回可用的 rot.json 路径：
      1) 若 rot_dir/<cat>/<rot_filename> 存在且未要求 remerge → 直接用
      2) 否则扫描 rot_dir/<cat>/<merge_glob> 合并
         - persist=True：写到 rot_dir/<cat>/<rot_filename>
         - persist=False：写临时文件
      3) 若 rot_dir 未给，则回退 --rot_json
    """
    rot_filename = getattr(args, "rot_filename", "final_shape_rotations.json")
    merge_glob = merge_glob or getattr(args, "rot_merge_glob", "cluster_*_rotations.json")
    dup_policy = dup_policy or getattr(args, "rot_dup_policy", "prefer_last")
    persist = persist if persist is not None else getattr(args, "persist_merged_rot", True)

    # 优先 rot_dir
    if getattr(args, "rot_dir", None):
        cat_dir = Path(args.rot_dir).resolve() / category
        direct = cat_dir / rot_filename
        if direct.exists() and not force_remerge:
            return direct.resolve()

        if merge_glob:
            cands = sorted(cat_dir.glob(merge_glob))
            if not cands:
                # 没有 cluster_*，若有 direct 就用；否则报错
                if direct.exists():
                    return direct.resolve()
                raise FileNotFoundError(f"{category}: 未找到 {merge_glob} 或 {rot_filename}")
        else:
            cands = []

        merged = _merge_rot_maps(cands, dup_policy=dup_policy or "prefer_last")
        out_obj = {
            "export_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "source_files": [p.name for p in cands],
                "dup_policy": dup_policy,
                "total_shapes": len(merged),
            },
            "shape_rotations": merged,
        }

        if persist:
            # 写回最终文件（真实生成 final_shape_rotations.json）
            cat_dir.mkdir(parents=True, exist_ok=True)
            with open(direct, "w") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            print(f"[INFO] {category}: 合并 {len(cands)} 个 cluster json → {direct}")
            return direct.resolve()
        else:
            # 临时文件
            tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix=f"rot_{category}_", delete=False)
            json.dump(out_obj, tf, ensure_ascii=False)
            tf.flush(); tf.close()
            print(f"[INFO] {category}: 合并 {len(cands)} 个 cluster json → 临时 {tf.name}")
            return Path(tf.name)

    # 回退单文件
    if getattr(args, "rot_json", None):
        fp = Path(args.rot_json).resolve()
        if fp.exists():
            return fp
    raise FileNotFoundError(f"{category}: 没有 rot_dir 也没有 rot_json")
# --- helper: case-insensitive pick (可复用你之前的版本) ---
def _ci_pick(parent: Path, name: str) -> Path | None:
    cand = parent / name
    if cand.exists():
        return cand
    norm = " ".join(name.split()).lower()
    for p in parent.iterdir():
        if p.is_file() and (" ".join(p.name.split()).lower() == norm or p.name.lower() == name.lower()):
            return p
    return None

# --- helper: 安全落盘（优先硬链，失败复制；相对 symlink 仅在显式要求时用） ---
def _place_into_raw(src: Path, raw_dir: Path, prefer: str = "hardlink") -> Path:
    """
    将 src '实体化' 到 raw_dir 下（文件名保持 basename）：
    - prefer=hardlink：同盘用硬链；跨盘/权限失败自动回退 copy2
    - prefer=copy    ：总是复制
    - prefer=symlink ：尽量软链；失败回退 copy2（不推荐）
    返回目标路径 dst。
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    dst = raw_dir / src.name

    # 若目标已是可用文件，直接返回；若是坏链接/空文件，先清掉
    try:
        if dst.is_file() and dst.stat().st_size > 0:
            return dst
        if dst.is_symlink() or dst.exists():
            try: dst.unlink()
            except IsADirectoryError: shutil.rmtree(dst, ignore_errors=True)
    except Exception:
        pass

    def _same_device(a: Path, b: Path) -> bool:
        try:
            return a.stat().st_dev == b.stat().st_dev
        except Exception:
            return False

    mode = (prefer or "hardlink").lower()

    # hardlink
    if mode == "hardlink":
        if _same_device(src, raw_dir):
            try:
                os.link(src, dst)
                return dst
            except Exception:
                pass  # 回退复制
        # 跨盘或失败：复制
        shutil.copy2(src, dst)
        return dst

    # copy
    if mode == "copy":
        shutil.copy2(src, dst)
        return dst

    # symlink（不推荐）
    if mode == "symlink":
        try:
            os.symlink(src, dst)  # 可改为 src.resolve() 取得绝对路径
            return dst
        except Exception:
            shutil.copy2(src, dst)
            return dst

    # 未知模式：按 hardlink 逻辑
    if _same_device(src, raw_dir):
        try:
            os.link(src, dst)
            return dst
        except Exception:
            pass

    shutil.copy2(src, dst)
    return dst

@dataclass
class PipelinePaths:
    root: Path
    downloads: Path
    aligned_glb: Path
    renders: Path
#     logs: Path
#     manifests: Path

    @staticmethod
    def make(root: Union[str, Path], downloads: Optional[Union[str, Path]] = None, dl_backend: Optional[str]=None) -> "PipelinePaths":
        root = Path(root)
        dl_path = Path(downloads) if downloads is not None else (root / "G-objaverse")
        dl = ensure_dir(dl_path)
        return PipelinePaths(
            root=root,
            downloads=dl,
            aligned_glb=ensure_dir(root / "aligned_glb") if dl_backend != "oxl" else ensure_dir(dl_path / OXL_OUT_CAT),
            renders=ensure_dir(root / "renders") if dl_backend != "oxl" else ensure_dir(dl_path / "renders_xl"),
        #     logs=ensure_dir(root / "logs"),
        #     manifests=ensure_dir(root / "manifests"),
        )


def step_download_assets(args, P: PipelinePaths, cat: Optional[List[str]] = None, inferred_cats: Optional[List[str]] = None) -> None:
    # """Download step dispatcher: custom(build_metadata.py) or oxl(ObjaverseXL)."""
    # if getattr(args, "dl_backend", "custom") == "oxl":
    #     # 检查是否使用per_category模式
    #     if getattr(args, "oxl_per_category_mode", False):
    #         if not cat or len(cat) != 1:
    #             raise ValueError("per_category OXL mode requires exactly one category")
    #         # 检查必需的参数
    #         if not getattr(args, 'label_dir', None) and not getattr(args, 'rot_dir', None):
    #             raise ValueError("per_category OXL mode requires --label_dir (or --rot_dir)")
    #         if not getattr(args, 'sha_index_csv', None):
    #             raise ValueError("per_category OXL mode requires --sha_index_csv")
    #         return step_download_assets_oxl_per_category(args, P, cat[0])
    #     else:
    #         return step_download_assets_oxl(args, P)

    # ---- custom downloader (existing behavior) ----
    """Run build_metadata.py to download/verify assets and produce metadata.csv.
    We pass through: --png_root(assests stored to, here default =/root), --maps_json, --categories, 
    --download_root (=downloads, default=png_root).
    If your build script also supports --ids_json / --ids_txt, include them when provided.
    Optionally, if rot_dir is provided and --ids_json not set, we will build an ids manifest
    by collecting shape ids from each category's rotation json and pass it as --ids_json.
    """
    build = Path(args.build_script).resolve()
    if not build.exists():
        raise FileNotFoundError(f"build_metadata.py not found at: {build}")

    if not args.maps_json:
        raise ValueError("No mapping JSON provided. Use --maps_json and/or --maps_dir.")
    
    if args.ids_json is None and getattr(args, "rot_dir", None):
        # categories to read ids from = inferred_cats if provided, else infer from rot_dir
        cats_for_ids = inferred_cats or infer_categories_from_rot_dir(Path(args.rot_dir), args.rot_filename)
        all_ids: List[str] = []
        if cats_for_ids:
            for cat in cats_for_ids:
                rot_fp = resolve_or_merge_rot_json(args, cat)
                all_ids.append(cat)  # 这里应该收集实际的shape ids，暂时用category名

    cmd = [sys.executable, str(build),
           "--png_root", str(Path(args.png_root).resolve()),
           "--maps_json", str(Path(args.maps_json).resolve()),
           "--ids_json", str(Path(args.rot_json).resolve()) if getattr(args, "rot_json", None) else str(rot_fp),
           "--download_root", str(P.downloads)]

    # Pass categories: prefer user-given else inferred from rot_dir (if available)
    if args.categories:
        cmd += ["--categories"] + [str(c) for c in args.categories]
    elif inferred_cats:
        cmd += ["--categories", str(inferred_cats[0])]

#     if args.ids_json is None:
#         cmd += ["--ids_json", str()]
#     elif args.ids_json:
#         cmd += ["--ids_json", str(Path(args.ids_json).resolve())]
    if args.ids_txt:
        cmd += ["--ids_txt", str(Path(args.ids_txt).resolve())]
    if args.debug_limit:
        cmd += ["--debug_limit", str(int(args.debug_limit))]

    if args.overwrite_download:
        cmd += ["--overwrite"]

    rc = sh(cmd, dry=args.dry_run)
    if rc != 0:
        sys.exit(rc)


# ---------- OXL download branch ----------

def step_download_assets_oxl_per_category(args, P: PipelinePaths, category: str) -> None:
    """
    按类别进行OXL下载的per_category模式：
    1. 从label_dir下搜索对应类别的label json和全部sha_index json进行筛选
    2. 下载zip到_stage目录
    3. 在label/sha下面的mesh里解压
    4. 先完整补全资产，再将3D资产移动到label/sha/raw文件夹
    
    参数:
    - args: 命令行参数
    - P: PipelinePaths对象
    - category: 当前处理的类别
    
    使用示例:
    # 方法1：使用独立的label_dir
    python mesh_pipeline_full.py \
        --dl_backend oxl \
        --label_dir /path/to/label_jsons \
        --sha_index_csv /path/to/all_sha_index.csv \
        --categories mug \
        --download_root /data/downloads \
        --root /data/pipeline \
        --png_root /data/png \
        --steps download
    
    # 方法2：复用现有的rot_dir（推荐）
    python mesh_pipeline_full.py \
        --dl_backend oxl \
        --rot_dir /path/to/rotation_jsons \
        --rot_filename final_shape_rotations.json \
        --sha_index_csv /path/to/all_sha_index.csv \
        --categories mug \
        --per_category_flow \
        --steps download align render
    
    注意：
    - label_dir下应该有对应类别的json文件，格式为 {"shape_rotations": {sha: {...}}}
    - sha_index_csv应该包含所有sha256的映射信息，CSV格式，必须包含sha256列
    - 通过sha256作为key进行关联筛选
    """
    try:
        from dataset_toolkits.trellis.datasets import ObjaverseXL as OXL
    except Exception as e:
        raise ImportError(f"Failed to import trellis.datasets.ObjaverseXL: {e}")

    # 1. 加载对应类别的label json和全部sha_index csv
    if not getattr(args, 'label_dir', None):
        raise ValueError("per_category mode requires --label_dir (or use existing --rot_dir)")
    if not getattr(args, 'sha_index_csv', None):
        raise ValueError("per_category mode requires --sha_index_csv")
    
    # 使用label_dir或rot_dir作为label json的根目录
    label_root = getattr(args, 'label_dir', None) or getattr(args, 'rot_dir', None)
    if not label_root:
        raise ValueError("Neither --label_dir nor --rot_dir provided")
    label_root = Path(label_root).resolve()
    rot_filename = getattr(args, 'rot_filename', 'final_shape_rotations.json')
    
    # 构建对应类别的label json路径
    label_json_path = label_root / category / rot_filename
    sha_index_csv_path = Path(args.sha_index_csv)
    
    if not label_json_path.exists():
        raise FileNotFoundError(f"Label json not found for category '{category}': {label_json_path}")
    if not sha_index_csv_path.exists():
        raise FileNotFoundError(f"SHA index CSV not found: {sha_index_csv_path}")
    
    # 加载数据
    import json as _json
    import pandas as _pd
    
    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = _json.load(f)
    
    # 读取CSV格式的sha_index
    sha_index_df = _pd.read_csv(sha_index_csv_path)
    
    print(f"[OXL-PER-CAT] Processing category: {category}")
    print(f"[OXL-PER-CAT] Label json: {label_json_path}")
    print(f"[OXL-PER-CAT] SHA index CSV: {sha_index_csv_path}")
    
    # 2. 筛选出当前category需要的sha256
    # 从label_data中提取该category的所有sha256
    category_shas = set()
    if isinstance(label_data, dict) and "shape_rotations" in label_data:
        # 新格式：{"shape_rotations": {sha: {...}}}
        rotations_data = label_data["shape_rotations"]
    else:
        # 旧格式：直接是sha到info的映射
        rotations_data = label_data
    
    for sha, info in rotations_data.items():
        if isinstance(info, dict):
            labels = info.get('label', [])
            if isinstance(labels, str):
                labels = [labels]
            labels = [str(x).strip() for x in labels if str(x).strip()]
            if category in labels:
                category_shas.add(sha)
    
    print(f"[OXL-PER-CAT] Found {len(category_shas)} SHAs for category '{category}'")
    
    # 3. 从sha_index_df中筛选出真正需要下载的条目
    # 筛选出在category_shas中的sha256行
    if 'sha256' not in sha_index_df.columns:
        raise ValueError(f"SHA index CSV must contain 'sha256' column: {sha_index_csv_path}")
    
    # 筛选出当前类别需要的sha256
    filtered_df = sha_index_df[sha_index_df['sha256'].isin(category_shas)]
    
    print(f"[OXL-PER-CAT] {len(filtered_df)} entries to download")
    
    if len(filtered_df) == 0:
        print(f"[OXL-PER-CAT] No entries to download for category '{category}', skipping...")
        return
    
    # 4. 构建metadata DataFrame（直接使用筛选后的数据）
    meta = filtered_df.copy()
    
    # 确保必要的列存在
    required_columns = ['sha256', 'source_url', 'local_path']
    for col in required_columns:
        if col not in meta.columns:
            if col == 'source_url':
                meta[col] = ''
            elif col == 'local_path':
                meta[col] = ''
            else:
                raise ValueError(f"Required column '{col}' missing from SHA index CSV")
    
    # 5. 设置输出目录结构
    out_cat = getattr(args, 'oxl_out_category', 'oxl')
    cat_dir = ensure_dir(P.downloads / out_cat)
    stage_dir = args.oxl_stage_dir if getattr(args, 'oxl_stage_dir', None) else (Path(cat_dir) / category / f"{category}_stage")
    stage_dir = ensure_dir(stage_dir)
    
    # 强制重拉zip：清空_stage
    if getattr(args, 'overwrite_download_repo', False):
        try:
            shutil.rmtree(stage_dir, ignore_errors=True)
        except Exception:
            pass
        stage_dir.mkdir(parents=True, exist_ok=True)
    
    # 6. 执行下载
    print(f"[OXL-PER-CAT] Downloading {len(meta)} objects to staging...")
    with github_auth_env(args):
        df = OXL.download(meta, str(stage_dir))
    
    if df is None or len(df) == 0:
        print(f"[OXL-PER-CAT] No objects downloaded for category '{category}'")
        return
    
    # 7. 处理下载的文件：解压、提取mesh、补全资产
    label_dir = ensure_dir(cat_dir / category)
    processed_count = 0
    
    oxl_reports: list[dict[str, Any]] = []

    for _, rec in df.iterrows():
        sha = rec.get('sha256')
        rel = rec.get('local_path', '')
        rel_str = str(rel)
        report_entry: Dict[str, Any] = {
            'sha': sha,
            'status': 'pending',
            'missing_textures': '',
            'missing_reason': '',
            'repo_missing': '',
            'repo_failed': '',
        }

        # 源文件路径
        obj_path = (stage_dir / rel).resolve()
        
        # 目标目录：label/sha
        inst_root = ensure_dir(label_dir / sha)
        dst_raw = ensure_dir(inst_root / 'raw')
        
        # 处理GitHub repo格式
        is_github_repo = str(rel).startswith('raw/github/repos/')
        zip_in_repo = None
        inner_repo_path = None
        
        if is_github_repo:
            parts = str(rel).split('/')
            try:
                zi = next(i for i, p in enumerate(parts) if p.endswith('.zip'))
                zip_rel = "/".join(parts[:zi+1])
                inner_repo_path = "/".join(parts[zi+1:]) if len(parts) > zi+1 else ''
                zip_in_repo = (stage_dir / zip_rel).resolve()
            except StopIteration:
                pass
        
        # 复制源文件到raw目录
        if is_github_repo and zip_in_repo and zip_in_repo.exists():
            obj_path = zip_in_repo
        elif obj_path.exists():
            target = dst_raw / obj_path.name
            if obj_path != target:
                try:
                    shutil.copy2(obj_path, target)
                    obj_path = target
                except Exception:
                    pass
        
        if not obj_path.exists() and not (is_github_repo and zip_in_repo and zip_in_repo.exists()):
            print(f"[OXL-PER-CAT] ✘ staging file missing for {sha}: {rel_str}")
            report_entry['status'] = 'stage_missing'
            oxl_reports.append(report_entry)
            if 'stats' in locals():
                stats.setdefault('stage_missing', 0)
                stats['stage_missing'] += 1
            continue

        # 解压到mesh目录
        mesh_path = None
        mesh_dir = None
        if _is_archive_name(obj_path.name):
            mesh_dir = ensure_dir(inst_root / 'mesh')
            try:
                _safe_extract_archive(obj_path, mesh_dir)
            except Exception as e:
                print(f"[OXL-PER-CAT] ✘ extract failed: {obj_path.name} -> {e}")
                continue
            
            # 查找mesh文件
            if is_github_repo and inner_repo_path:
                cand = (mesh_dir / inner_repo_path).resolve()
                if cand.exists():
                    mesh_path = cand if cand.is_file() else _pick_mesh_file(cand)
                else:
                    mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(inst_root)
            else:
                mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(inst_root)
            
            # 可选：删除原始压缩包
            if mesh_path and getattr(args, 'oxl_prune_zips', False):
                try:
                    os.remove(obj_path)
                except Exception:
                    pass
        else:
            if obj_path.suffix.lower() in ['.glb','.gltf','.obj','.fbx','.stl','.ply','.blend']:
                mesh_path = obj_path
                mesh_dir = obj_path.parent
        
        if not mesh_path or not mesh_path.exists():
            print(f"[OXL-PER-CAT] ✘ no mesh found for {sha}")
            if 'stats' in locals():
                stats["extract_failed"] += 1
            report_entry['status'] = 'no_mesh'
            oxl_reports.append(report_entry)
            continue
        
        # 先补全资产（在原始位置）
        try:
            from download_toolkits.build_metadata import ensure_complete_asset_anyformat
            
            # 使用原始mesh目录作为local_roots，保持目录结构
            # _local_roots = [p for p in [mesh_dir, inst_root] if p and p.exists()]
            # _local_roots: list[Path] = []    
            # for base in [mesh_dir, inst_root, getattr(mesh_path, "parent", None)]:
            #     if base and Path(base).exists():
            #         _local_roots.append(Path(base).resolve())
            # _seen=set(); _local_roots=[p for p in _local_roots if str(p) not in _seen and not _seen.add(str(p))]
            _local_roots: list[Path] = []
            for base in (mesh_dir, inst_root):
                if isinstance(base, Path) and base.exists():
                    _local_roots.append(base.resolve())

            # 2) 加入 OBJ 的真实父目录（新方案的改进点）
            try:
                parent = getattr(mesh_path, "parent", None)
                if isinstance(parent, Path) and parent.exists():
                    _local_roots.append(parent.resolve())
            except Exception:
                pass

            # 3) 去重（按字符串，保持顺序）
            _seen = set()
            _local_roots = [p for p in _local_roots if (s := str(p)) not in _seen and not _seen.add(s)]
            # 确保mesh_dir不为None
            target_raw_dir = dst_raw if dst_raw.exists() else mesh_dir
            rep = ensure_complete_asset_anyformat(
                asset_path = mesh_path,
                src_url    = rec.get('source_url','') or '',
                raw_dir    = target_raw_dir,  # 先在mesh目录补全
                overwrite  = getattr(args, 'overwrite_consolidate', False),
                local_only = True,
                local_roots= _local_roots,
            )
            
            # 输出摘要
            kind = rep.get('kind','?')
            dl = rep.get('downloaded',0)
            miss = rep.get('missing_names',[]) or rep.get('missing',[])
            repo_missing = rep.get('repo_missing_names', []) or []
            repo_failed = rep.get('found_but_failed', []) or []
            missing_reason_map = rep.get('missing_reason_map', {}) or {}
            local_missing = [name for name, reason in missing_reason_map.items() if reason == 'local_missing']
            repo_missing_reason = [name for name, reason in missing_reason_map.items() if reason == 'repo_missing']
            other_missing = {name: reason for name, reason in missing_reason_map.items()
                             if reason not in ('local_missing', 'repo_missing')}

            print(f"[OXL-PER-CAT] [{sha}] ensure({kind}): downloaded={dl} missing={len(miss)} repo_missing={len(repo_missing)} repo_failed={len(repo_failed)}")
            if repo_missing:
                print("               · repo_missing examples: " + ", ".join(repo_missing[:5]))
            if repo_failed:
                print("               · repo_failed examples: " + ", ".join(repo_failed[:5]))
            if local_missing:
                print("               · local_missing examples: " + ", ".join(local_missing[:5]))
            if other_missing:
                sample = list(other_missing.items())[:5]
                print("               · other_missing: " + ", ".join(f"{k}:{v}" for k,v in sample))
            if 'stats' not in locals():
                stats = {"total_in_meta": len(meta), "downloaded_rows": 0,
                        "ensure_counts": {"obj": {"downloaded":0,"missing":0},
                                        "fbx": {"downloaded":0,"missing":0},
                                        "glb": {"downloaded":0,"missing":0}},
                        "skipped_existing": 0, "extract_failed": 0}
            # 类型归一（rep.get('kind') 返回 obj/fbx/glb/…）
            _kind = (rep.get('kind','?') or '?').lower()
            if _kind not in stats["ensure_counts"]:
                stats["ensure_counts"][_kind] = {"downloaded":0,"missing":0}
            stats["ensure_counts"][_kind]["downloaded"] += int(rep.get('downloaded',0) or 0)
            stats["ensure_counts"][_kind]["missing"]    += int(len(rep.get('missing_names',[]) or rep.get('missing',[]) or []))
            stats["ensure_counts"][_kind].setdefault("repo_missing", 0)
            stats["ensure_counts"][_kind]["repo_missing"] += len(repo_missing)
            stats["ensure_counts"][_kind].setdefault("local_missing", 0)
            stats["ensure_counts"][_kind]["local_missing"] += len(local_missing)
            stats["ensure_counts"][_kind].setdefault("repo_failed", 0)
            stats["ensure_counts"][_kind]["repo_failed"] += len(repo_failed)
            stats["downloaded_rows"] += 1

            status_val = 'fetched'
            if miss:
                if repo_missing or repo_missing_reason:
                    status_val = 'repo_missing'
                elif local_missing:
                    status_val = 'local_missing'
                else:
                    status_val = 'missing'
            report_entry.update({
                'status': status_val,
                'missing_textures': ";".join(miss),
                'missing_reason': ";".join(f"{k}:{v}" for k,v in missing_reason_map.items()),
                'repo_missing': ";".join(repo_missing),
                'repo_failed': ";".join(repo_failed),
            })
        except Exception as e:
            print(f"[OXL-PER-CAT] ensure assets failed: {e}")
            if 'stats' in locals():
                stats["extract_failed"] += 1
            report_entry['status'] = 'ensure_failed'
            report_entry['missing_reason'] = str(e)
            oxl_reports.append(report_entry)
            continue

        # 补全完成后，将mesh文件移动到raw目录
        final_mesh_path = _place_into_raw(mesh_path, dst_raw, prefer=getattr(args, "oxl_consolidate_link_mode", "hardlink"))
        
        # 可选：清理mesh目录
        if getattr(args, 'oxl_prune_mesh_dir', False):
            try:
                shutil.rmtree(inst_root / 'mesh', ignore_errors=True)
            except Exception:
                pass
        
        processed_count += 1
        if report_entry['status'] == 'pending':
            report_entry['status'] = 'fetched' if not miss else report_entry['status']
        oxl_reports.append(report_entry)

    if oxl_reports:
        total = len(oxl_reports)
        status_counts: Dict[str, int] = {}
        for entry in oxl_reports:
            status = entry.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        print("[OXL-PER-CAT] ensure summary:")
        print(f"  · total items: {total}")
        for key in sorted(status_counts.keys()):
            print(f"  · {key}: {status_counts[key]}")
    
    print(f"[OXL-PER-CAT] Processed {processed_count}/{len(df)} objects for category '{category}'")
    
    # 8. 生成metadata.csv
    meta_csv = label_dir / f"metadata_{category}.csv"
    import csv
    fieldnames = ["sha256","local_path","category","shape_key","source_url","rendered"]
    
    with meta_csv.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        wr.writeheader()
        for _, rec in df.iterrows():
            sha = rec.get('sha256')
            # 查找最终的mesh文件位置
            raw_dir = label_dir / sha / 'raw'
            actual_mesh = _pick_mesh_file(raw_dir) if raw_dir.exists() else None
            wr.writerow({
                'sha256': sha,
                'local_path': str(actual_mesh.resolve()) if actual_mesh else '',
                'category': category,
                'shape_key': sha,
                'source_url': rec.get('source_url', ''),
                'rendered': 'False',
            })
    
    print(f"[OXL-PER-CAT] Wrote metadata: {meta_csv} ({processed_count} rows)")
    import csv as _csv
    import json as _json
    if 'stats' not in locals():
        stats = {"total_in_meta": len(meta), "downloaded_rows": int(processed_count),
                "ensure_counts": {}, "skipped_existing": 0, "extract_failed": 0}

    report_json = {
        "label": category,
        "total_in_meta": int(stats.get("total_in_meta", 0)),
        "processed_rows": int(processed_count),
        "extract_failed": int(stats.get("extract_failed", 0)),
        "downloaded_rows": int(stats.get("downloaded_rows", 0)),
        "ensure_counts": stats.get("ensure_counts", {}),
        "stage_dir": str(stage_dir),
        "label_dir": str(label_dir),
        "metadata_csv": str(meta_csv),
    }

    (label_dir / "report.json").write_text(_json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OXL-PER-CAT] Wrote report: {label_dir/'report.json'}")

    # 同步一份 CSV（便于表格审阅）
    rep_csv = label_dir / "report.csv"
    with rep_csv.open("w", newline="", encoding="utf-8") as f:
        wr = _csv.writer(f)
        wr.writerow(["label","total_in_meta","processed_rows","downloaded_rows","extract_failed","metadata_csv"])
        wr.writerow([category, report_json["total_in_meta"], report_json["processed_rows"],
                    report_json["downloaded_rows"], report_json["extract_failed"], report_json["metadata_csv"]])
        # 展开 ensure_counts
        wr.writerow([])
        wr.writerow(["kind","ensure_downloaded","ensure_missing"])
        for k, v in report_json["ensure_counts"].items():
            wr.writerow([k, v.get("downloaded",0), v.get("missing",0)])
    print(f"[OXL-PER-CAT] Wrote report CSV: {rep_csv}")
    # 9. 清理staging目录
    if stage_dir.exists() and not getattr(args, 'oxl_keep_stage', False):
        try:
            shutil.rmtree(stage_dir)
            print(f"[OXL-PER-CAT] Cleaned staging directory: {stage_dir}")
        except Exception:
            pass

"""
def step_download_assets_oxl(args, P: PipelinePaths) -> None:
    #Use trellis/datasets/ObjaverseXL to download objects, extract, pick mesh, and write metadata.csv.
    try:
        from dataset_toolkits.trellis.datasets import ObjaverseXL as OXL
    except Exception as e:
        raise ImportError(f"Failed to import trellis.datasets.ObjaverseXL: {e}")

    out_cat = getattr(args, 'oxl_out_category', 'oxl')
    cat_dir = ensure_dir(P.downloads / out_cat)
    meta_csv = _oxl_metadata_csv_path(args)
    reuse_meta = (not getattr(args, "oxl_refresh_metadata", False) and _has_complete_metadata_csv(meta_csv)) and not getattr(args, "overwrite_download_repo", False)   # 不强制重下
    if reuse_meta:
        print(f"[OXL] metadata.csv found ({meta_csv}); skip fetching metadata.")
        # 若你需要本轮继续"归拢/ensure"，走我们已有的兜底重归拢逻辑
        # df = None  # 模拟"本次没有新条目"
        try:
            import pandas as _pd
        except ImportError:
            raise ImportError("pandas is required for OXL mode")
        meta = _pd.read_csv(meta_csv)
    else:
        # 原来的获取元数据和 download_objects 逻辑保持不变
        print(f"[OXL] Fetching metadata from source={args.oxl_source} ...")
        meta = OXL.get_metadata(args.oxl_source)
    
    prefer_label_root = bool(getattr(args, 'oxl_prefer_label_root', False))
    # Prepare staging directory for raw downloads
     # Prepare staging directory for raw downloads
    stage_dir = args.oxl_stage_dir if getattr(args, 'oxl_stage_dir', None) else (cat_dir / '_stage')
    stage_dir = ensure_dir(stage_dir)

    # NEW: 强制重拉 zip：清空 _stage
    if getattr(args, 'overwrite_download_repo', False):
        try:
            shutil.rmtree(stage_dir, ignore_errors=True)
        except Exception:
            pass
        stage_dir.mkdir(parents=True, exist_ok=True)

    stage_dir = ensure_dir(stage_dir)

    # Resume: if final targets already exist and not overwriting, pre-filter metadata to avoid re-downloading
    if not getattr(args, 'overwrite_download', False):
        try:
            try:
                import pandas as _pd
            except ImportError:
                _pd = None
            if _pd and isinstance(meta, _pd.DataFrame) and 'sha256' in meta.columns:
                skip_shas = []
                for d in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
                    if (d / 'raw').is_dir():
                        # consider as done if raw contains any file
                        try:
                            if any((d / 'raw').iterdir()):
                                skip_shas.append(d.name)
                        except Exception:
                            pass
                if skip_shas:
                    before = len(meta)
                    meta = meta[~meta['sha256'].isin(skip_shas)]
                    after = len(meta)
                    if before != after:
                        print(f"[OXL] Resume: skip {before - after} already consolidated instances")
        except Exception:
            pass

    print("[OXL] Downloading objects to staging (with optional label filtering)...")
    # NOTE: We pass output_dir as staging dir parent to keep raw placement under _stage
    # ---- pre-filter by labels BEFORE download (optional) ----
    if args.label_json and Path(args.label_json).exists() and getattr(args, 'labels', None):
        import json as _json
        label_map = _json.loads(Path(args.label_json).read_text(encoding='utf-8'))
        targets = _norm_label_list(getattr(args, "labels", None))
        mode = getattr(args, 'label_match_mode', 'any')

        def _has_labels(sha: str) -> bool:
            entry = label_map.get(sha) or {}
            labs = entry.get('label')
            if isinstance(labs, str):
                labs = [labs]
            labs = [str(x).strip() for x in (labs or []) if str(x).strip()]
            if not labs:
                return False
            return set(targets).issubset(set(labs)) if mode == 'all' else any(l in labs for l in targets)

        before = len(meta)
        meta = meta[meta['sha256'].apply(_has_labels)]
    print(f"[OXL] Label pre-filter: {before} -> {len(meta)} by labels={targets} mode={mode}")

    df = OXL.download(meta, str(stage_dir), label_json=str(args.label_json) if args.label_json else None,
                      labels=args.labels, label_match_mode=getattr(args, 'label_match_mode', 'any'))
    # # 如果 OXL 没返回新条目，但 _stage 里有 zip，可选择从 _stage 重建 df 做二次归拢, only number of zips, but not as all the items
    #     import pandas as _pd
    # except Exception:
    #     _pd = None

    # if (df is None or (hasattr(df, '__len__') and len(df)==0)) and getattr(args, 'oxl_force_reconsolidate', False):
    #     zips = list(stage_dir.rglob("*.zip"))
    #     if zips and _pd is not None:
    #         recs = []
    #         for z in zips:
    #             # 约定：sha256 用 zip 文件名（或留空）；local_path 用相对 _stage 的路径
    #             try:
    #                 rel = z.relative_to(stage_dir).as_posix()
    #             except Exception:
    #                 rel = z.name
    #             recs.append({"sha256": z.stem, "local_path": rel, "source_url": ""})
    #         df = _pd.DataFrame(recs)
    #         print(f"[OXL] Reconsolidate from _stage: {len(df)} zip(s) found.")

    # 若提供 label_json，则载入以便进行分组归档
    label_map = {}
    target_labels = []
    match_mode = getattr(args, 'label_match_mode', 'any')
    if args.label_json and Path(args.label_json).exists():
        try:
            import json as _json
            label_map = _json.loads(Path(args.label_json).read_text(encoding='utf-8'))
            if args.labels:
                target_labels = [s.strip() for s in str(args.labels).split(',') if s.strip()]
        except Exception:
            label_map = {}

    # label -> rows 收集；若无 label 则使用 out_cat 作为聚合目录
    label_to_rows: Dict[str, list] = {}

    oxl_reports: list[dict[str, Any]] = []

    for rec in df.to_dict('records'):
        sha = rec.get('sha256')
        rel = rec.get('local_path') or ''
        rel_str = str(rel)
        report_entry: Dict[str, Any] = {
            'sha': sha,
            'status': 'pending',
            'missing_textures': '',
            'missing_reason': '',
            'repo_missing': '',
            'repo_failed': '',
        }
        if not rel_str:
            print(f"[OXL] ✘ manifest missing for {sha}: empty local_path")
            report_entry['status'] = 'manifest_missing'
            oxl_reports.append(report_entry)
            if 'stats' in locals():
                stats.setdefault('manifest_missing', 0)
                stats['manifest_missing'] += 1
            continue
        # source path from staging area (can be a path inside repo zip for github source)
        obj_path = (stage_dir / rel).resolve()

        # 实体直接落地到 <out_cat>/<sha>/raw 下
        # 若优先 label 根：先不建规范位，延后按 label 再决定根
        # if prefer_label_root:
        #     inst_root = ensure_dir(cat_dir / '_pool' / sha)  # 池子目录，给解压/挑 mesh/补全用
        # else:
        #     inst_root = ensure_dir(cat_dir / sha)
        #     dst_raw = ensure_dir(inst_root / 'raw')
        # 在 per_category_flow 场景，首次解包根改为 label 根（在 label 循环里设 _inst_root）
        inst_root = ensure_dir(cat_dir / sha)   # 占位（不再建 _pool）
        dst_raw = ensure_dir(inst_root / 'raw')

        # Special handling: github repos use local_path like
        #   raw/github/repos/<user>/<repo>/<sha>/<path/inside/repo>
        # The actual zip lives at:
        #   raw/github/repos/<user>/<repo>/<sha>.zip
        # ---- robust GitHub repo.zip handling (OXL format) ----
        # ---- robust GitHub repo.zip handling (OXL official format) ----
        is_github_repo = rel_str.startswith('raw/github/repos/')
        zip_in_repo = None
        inner_repo_path = None
        if is_github_repo:
            parts = rel_str.split('/')
            # 官方格式：raw/github/repos/<org>/<repo>.zip/<inner/...>
            try:
                zi = next(i for i, p in enumerate(parts) if p.endswith('.zip'))
                zip_rel = "/".join(parts[:zi+1])                       # .../<repo>.zip
                inner_repo_path = "/".join(parts[zi+1:]) if len(parts) > zi+1 else ''
                zip_in_repo = (stage_dir / zip_rel).resolve()
            except StopIteration:
                # 罕见：非 zip 布局（save_repo_format='files'）
                zip_in_repo = None
                inner_repo_path = "/".join(parts[6:]) if len(parts) > 6 else ''

        try:
            # if is_github_repo and zip_in_repo and zip_in_repo.exists():
            #     target = dst_raw / zip_in_repo.name
            #     if zip_in_repo != target:
            #         ensure_dir(target.parent)
            #         try:
            #             shutil.move(str(zip_in_repo), str(target))
            #         except Exception:
            #             shutil.copy2(zip_in_repo, target)
            #             try:
            #                 os.remove(zip_in_repo)
            #             except Exception:
            #                 pass
            #     obj_path = target
            if is_github_repo and zip_in_repo and zip_in_repo.exists():
                obj_path = zip_in_repo
            elif obj_path.exists():
                target = dst_raw / obj_path.name
                if obj_path != target:
                    ensure_dir(target.parent)
                    # move from staging to final to avoid duplicate storage
                    try:
                        shutil.move(str(obj_path), str(target))
                    except Exception:
                        # fallback to copy if move fails (e.g., cross-device)
                        shutil.copy2(obj_path, target)
                        try:
                            os.remove(obj_path)
                        except Exception:
                            pass
                obj_path = target
        except Exception:
            pass

        mesh_path: Optional[Path] = None
        if _is_archive_name(obj_path.name):
            mesh_dir = ensure_dir(inst_root / 'mesh')
            try:
                _safe_extract_archive(obj_path, mesh_dir)
            except Exception as e:
                print(f"[OXL] ✘ extract failed: {obj_path.name} -> {e}")
                continue
            # Prefer the expected inner path if provided (github repo case)
            if is_github_repo and inner_repo_path:
                cand = (mesh_dir / inner_repo_path).resolve()
                if cand.exists():
                    mesh_path = cand if cand.is_file() else _pick_mesh_file(cand)
                else:
                    mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(inst_root)
            else:
                mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(inst_root)
            # optional: prune original archive after successful extraction
            if mesh_path and getattr(args, 'oxl_prune_zips', False):
                try:
                    os.remove(obj_path)
                except Exception:
                    pass
        else:
            if obj_path.suffix.lower() in ['.glb','.gltf','.obj','.fbx','.stl','.ply','.blend']:
                mesh_path = obj_path

        if not mesh_path or not mesh_path.exists():
            print(f"[OXL] ✘ no mesh found for {sha}")
            continue

        # 计算该 sha 应归属的 label 列表
        labels_for_sha = []
        if label_map:
            entry = label_map.get(sha) or {}
            labs = entry.get('label')
            if isinstance(labs, str):
                labs = [labs]
            labs = [str(x).strip() for x in (labs or []) if str(x).strip()]
            if target_labels:
                item_set = set(labs)
                tgt_set = set(target_labels)
                if match_mode == 'all':
                    # 满足 all 的也应当至少属于这些标签之一，分发到交集中
                    if tgt_set.issubset(item_set):
                        labels_for_sha = [l for l in labs if l in tgt_set]
                else:
                    inter = [l for l in labs if l in tgt_set]
                    labels_for_sha = inter
            else:
                labels_for_sha = labs
        # 计算 labels_for_sha 之后、进入写入前：
        final_label_list = labels_for_sha or [out_cat]  # 没 label 就归到 out_cat
        for lab in final_label_list:
            lab_safe = lab.replace('/', '_')
            # 若优先 label 根，实体根 = <cat_dir>/<lab>/<sha>
            _inst_root = ensure_dir(( cat_dir / lab_safe) / sha) if prefer_label_root else inst_root
            # NEW: consolidate 粒度覆写：清理目标 sha 根目录
            if getattr(args, 'overwrite_consolidate', False):
                try:
                    shutil.rmtree(_inst_root, ignore_errors=True)
                except Exception:
                    pass
                _inst_root = ensure_dir(_inst_root)  # 重新创建
            _dst_raw = ensure_dir(_inst_root / 'raw')


            # 1) 把 staging 里的源（zip 或文件）准备成 obj_path（不要移动 zip，本质只用它来解）
            obj_path = (stage_dir / rel).resolve()
            is_github_repo = rel_str.startswith('raw/github/repos/')
            zip_in_repo = None
            inner_repo_path = None
            if is_github_repo:
                parts = rel_str.split('/')
                # 官方格式：raw/github/repos/<org>/<repo>.zip/<inner/...>
                try:
                    zi = next(i for i, p in enumerate(parts) if p.endswith('.zip'))
                    zip_rel = "/".join(parts[:zi+1])
                    inner_repo_path = "/".join(parts[zi+1:]) if len(parts) > zi+1 else ''
                    zip_in_repo = (stage_dir / zip_rel).resolve()
                except StopIteration:
                    pass
            if is_github_repo and zip_in_repo and zip_in_repo.exists():
                obj_path = zip_in_repo
            else:
                # 普通文件：复制到 _dst_raw 下便于留痕（可选）
                if obj_path.exists() and obj_path.parent != _dst_raw:
                    target = _dst_raw / obj_path.name
                    if not target.exists():
                        try:
                            shutil.copy2(obj_path, target)
                        except Exception:
                            pass
                    obj_path = target

            # 2) 解包到 _inst_root/mesh 并挑 mesh（你的原逻辑不变）
            # mesh_path = None
            # mesh_dir = ensure_dir(_inst_root / 'mesh')
            # if _is_archive_name(obj_path.name):
            #     try:
            #         _safe_extract_archive(obj_path, mesh_dir)
            #     except Exception as e:
            #         print(f"[OXL] ✘ extract failed: {obj_path.name} -> {e}")
            #         continue
            #     if is_github_repo and inner_repo_path:
            #         cand = (mesh_dir / inner_repo_path).resolve()
            #         mesh_path = cand if cand.is_file() else _pick_mesh_file(cand) if cand.exists() else None
            #     if not mesh_path:
            #         mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(_inst_root)
            # else:
            #     if obj_path.suffix.lower() in ['.glb','.gltf','.obj','.fbx','.stl','.ply','.blend']:
            #         # 非压缩直接放到 mesh_dir
            #         dst_mesh = mesh_dir / obj_path.name
            #         if not dst_mesh.exists():
            #             try:
            #                 shutil.copy2(obj_path, dst_mesh)
            #             except Exception:
            #                 dst_mesh = obj_path
            #         mesh_path = dst_mesh

            # if not mesh_path or not mesh_path.exists():
            #     print(f"[OXL] ✘ no mesh found for {sha}")
            #     continue
            # 2) 直接从“首次解压”的 inst_root/mesh 复用
            mesh_dir = ensure_dir(inst_root / 'mesh')  # 注意：这里用首次的 inst_root，而不是 _inst_root
            mesh_path = None
            if is_github_repo and inner_repo_path:
                cand = (mesh_dir / inner_repo_path).resolve()
                mesh_path = cand if cand.is_file() else (_pick_mesh_file(cand) if cand.exists() else None)
            if not mesh_path:
                mesh_path = _pick_mesh_file(mesh_dir) or _pick_mesh_file(inst_root)

            if not mesh_path or not mesh_path.exists():
                print(f"[OXL] ✘ no mesh found for {sha}")
                continue

            # 3) 写入 label 目录条目（这里我们“实体已在 label 根”，只需把 link 指向本地 mesh）
            label_dir = ensure_dir(cat_dir / lab_safe)
            inst_dir = ensure_dir(label_dir / sha / 'raw')
            link_path = _place_into_raw(mesh_path, inst_dir, prefer=getattr(args, "oxl_consolidate_link_mode", "hardlink"))
            # print(f"[OXL] consolidated: {link_path}")
            try:
                if not link_path.exists():
                    os.makedirs(link_path.parent, exist_ok=True)
                    try:
                        os.symlink(mesh_path, link_path)
                    except FileExistsError:
                        pass
            except Exception:
                # 硬链/复制兜底（你已经有相同逻辑，可复用）
                try:
                    if not link_path.exists():
                        os.makedirs(link_path.parent, exist_ok=True)
                        os.link(mesh_path, link_path)
                except Exception:
                    if not link_path.exists():
                        shutil.copy2(mesh_path, link_path)
            # --- NEW:补齐资产到 raw（本地优先，纯本地）---
            try:
                # try:
                #     # 本仓同目录使用
                #     from build_metadata import ensure_complete_asset_anyformat
                # except Exception:
                #     # 兼容包内路径（若你确实以包名安装）
                from dataset_toolkits.build_metadata import ensure_complete_asset_anyformat

                _local_roots = [p for p in [(_inst_root / 'mesh'), _inst_root] if p.exists()]
                target_asset = link_path if link_path.exists() else mesh_path
                rep = ensure_complete_asset_anyformat(
                    asset_path = target_asset,
                    src_url    = rec.get('source_url','') or '',
                    raw_dir    = inst_dir,                # <sha>/raw
                    overwrite  = getattr(args, 'overwrite_consolidate', False),
                    local_only = True,                    # 纯本地，不联网
                    local_roots= _local_roots,
                )
                # 统一摘要输出（带 sha / fbx 名 / roots）
                kind = rep.get('kind','?'); dl = rep.get('downloaded',0)
                miss = rep.get('missing_names',[]) or rep.get('missing',[])
                sat_local = rep.get('satisfied_local', None)
                blu = rep.get('tex_index_blender_count', None); rgx = rep.get('tex_index_regex_count', None)

                fbx_name = (target_asset.name if target_asset else mesh_path.name)
                roots_str = ", ".join(str(p) for p in _local_roots) if _local_roots else "(none)"
                print(f"[OXL] [{sha}] ensure({kind}) on {fbx_name}: downloaded={dl} missing={len(miss)}"
                    + (f" satisfied_local={sat_local}" if sat_local is not None else "")
                    + (f" index(b/r)={blu}/{rgx}" if (blu is not None and rgx is not None) else ""))
                print(f"       · search_roots: {roots_str}")
                if rep.get('fetched_names'):
                    print("       · placed:", ", ".join(rep['fetched_names'][:10]) + (" ..." if len(rep['fetched_names'])>10 else ""))
                if miss:
                    print("       · missing:", ", ".join([str(x) for x in miss[:10]]) + (" ..." if len(miss)>10 else ""))

            except Exception as e:
                print(f"[OXL] ensure assets (local) failed: {e}")


            # --- NEW: 可选清理 <sha>/mesh ---
            if getattr(args, 'oxl_prune_mesh_dir', False):
                try:
                    shutil.rmtree(_inst_root / 'mesh', ignore_errors=True)
                except Exception:
                    pass
            label_to_rows.setdefault(lab_safe, []).append({
                'sha256': sha,
                'local_path': str(link_path.resolve() if link_path.exists() else mesh_path.resolve()),
                'category': lab_safe,
                'shape_key': sha,
                'source_url': '',
                'rendered': 'False',
            })

    # 分别在各 label 目录写出 metadata.csv
    import csv
    total = 0
    for lab, rows in label_to_rows.items():
        lab_dir = cat_dir if lab == out_cat else (cat_dir / lab)
        meta_csv = lab_dir / 'metadata.csv'
        fieldnames = ["sha256","local_path","category","shape_key","source_url","rendered"]
        with meta_csv.open('w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            wr.writeheader()
            for row in rows:
                wr.writerow(row)
        print(f"[OXL] Wrote metadata: {meta_csv} (+{len(rows)} rows)")
        total += len(rows)
    if total == 0:
        print("[OXL] No usable meshes to record in metadata.csv")

    # Cleanup staging area if not requested to keep
    if stage_dir.exists() and not getattr(args, 'oxl_keep_stage', False):
        try:
            shutil.rmtree(stage_dir)
            print(f"[OXL] Cleaned staging directory: {stage_dir}")
        except Exception:
            pass
    # 新增：按需清理 <out_cat>/_pool（默认删除，--oxl_keep_pool 则保留）
    pool_dir = cat_dir / "_pool"
    if pool_dir.exists() and not getattr(args, 'oxl_keep_pool', False):
        try:
            shutil.rmtree(pool_dir)
            print(f"[OXL] Pruned pool directory: {pool_dir}")
        except Exception:
            pass       
"""
def step_align_export_one_category(args, P: PipelinePaths, category: str) -> Path:
    """Invoke Blender in background to align + export GLBs for a single category.
    Returns the output directory containing GLBs for that category.
    """
    blender = str(args.blender)
    align_script = Path(args.align_script).resolve()
    if not align_script.exists():
        raise FileNotFoundError(f"align_export_snapshot_blender.py not found: {align_script}")
    
    # >>> per-category rotation json <<<
    rot_fp = resolve_or_merge_rot_json(args, category)
    out_dir = ensure_dir(P.aligned_glb / category)
    # --- 若 dl_backend=oxl，重定向下载与分类根 ---
    if getattr(args, "dl_backend", "custom") == "oxl":
        if not getattr(args, "download_root", None):
            raise ValueError("--dl_backend oxl requires --download_root")
        DOWNLOAD_ROOT = Path(args.download_root).resolve()
        OXL_CAT   = str(getattr(args, "oxl_out_category", "oxl"))
        # 供后续步骤统一使用的“分类（label）根”
        OXL_LABELS_ROOT = DOWNLOAD_ROOT / OXL_CAT / category
        OXL_LABELS_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[OXL] downloads under: {OXL_LABELS_ROOT}")
        OXL_OUT_CAT = "aligned_xl"
        out_dir = ensure_dir(DOWNLOAD_ROOT / OXL_OUT_CAT / category)
    if getattr(args, "align_out_root", None) and args.dl_backend == "oxl":
        out_dir = Path(args.align_out_root).resolve() / category
    # Skip check: if aligned dir already has GLBs and not overwrite
    has_glb = any(out_dir.glob("*.glb"))
    has_obj = any(p.is_dir() and p.name.endswith("_obj") for p in out_dir.iterdir())
    if out_dir.exists() and not args.overwrite_align:
        if args.export_format == "glb":
            if any(out_dir.glob("*.glb")):
                print(f"[SKIP] Align for {category}: already has GLBs in {out_dir}")
                return out_dir
        else:
            if any(p.is_dir() and p.name.endswith("_obj") for p in out_dir.iterdir()):
                print(f"[SKIP] Align for {category}: already has OBJ folders in {out_dir}")
                return out_dir

    # Compose Blender background command; args after "--" are passed to the script.
    bl_cmd = [
        str(blender), "-b", "-P", str(align_script), "--",
        "--json", str(Path(args.rot_json).resolve()) if getattr(args, "rot_json", None) else str(rot_fp),
        "--data_root", str(Path(args.download_root).resolve() if args.dl_backend == "oxl" else P.downloads),
        "--category", category,
        "--out_dir", str(out_dir),
        "--snapshot", "snapshot.png",
    ]

    if args.normalize_unit:
        bl_cmd += ["--normalize_unit", "--target_size", str(float(args.target_size))]
    if args.prefer_gpu:
        bl_cmd += ["--prefer_gpu"]
    if args.topdown:
        bl_cmd += ["--topdown"]
    if args.clean_aligned_dir:
        bl_cmd += ["--clean_out_dir"]
    if args.overwrite_align:
        bl_cmd += ["--overwrite"]
    if args.limit:
        bl_cmd += ["--limit", str(int(args.limit))]
    if args.require_hollow_for:
        bl_cmd += ["--require_hollow_for", *args.require_hollow_for]
    if args.grid_n:
        bl_cmd += ["--grid_n",  str(int(args.grid_n))]
    if args.export_format:
        bl_cmd += ["--export_format", str(args.export_format)]
    # 新：是否传 OXL 路径参数
    if getattr(args, "dl_backend", "custom") == "oxl":
        bl_cmd += ["--oxl_layout", "--oxl_out_category", str(getattr(args, "oxl_out_category", "oxl"))]

    # 新：是否让 align 在脚本内按 label 过滤
    if getattr(args, "dl_backend", "custom") == "oxl" and getattr(args, "labels", None):
        bl_cmd += ["--oxl_filter_by_label"]
    rc = sh(bl_cmd, dry=args.dry_run)
    if rc != 0:
        sys.exit(rc)

    return out_dir


def iter_glbs(aligned_cat_dir: Path) -> Iterable[Path]:
    for p in sorted(aligned_cat_dir.glob("*.glb")):
        yield p

import tqdm
def step_render(args, P: PipelinePaths, models: Iterable[Path], category: str) -> None:
    """Render each GLB/OBJ using render_orbit_rgbd.py into the chosen per-shape directory."""
    render_script = Path(args.render_script).resolve()
    if not render_script.exists():
        raise FileNotFoundError(f"render_orbit_rgbd.py not found: {render_script}")
    models_list = list(models[:args.limit] if getattr(args, "limit", None) else models)
    total = len(models_list)
    if total == 0:
        print("[render] no models to render")
        return

    # ========== 批量模式：一次调用渲染脚本，输入指向对齐目录 ==========
    if getattr(args, "render_batch_mode", "single") == "batch":
        aligned_dir = Path(models_list[0]).parent
        out_root = (P.renders / category)
        out_root.mkdir(parents=True, exist_ok=True)

        print(f"[render/batch] cat={category}  dir={aligned_dir}  out={out_root}  total={total}")
        t0 = time.perf_counter()

        cmd = [sys.executable, str(render_script),
               "--obj", str(aligned_dir),
               "--out", str(out_root),
               "--num_cams", str(int(args.num_cams)),
               "--elev_deg", str(float(args.elev_deg)),
               "--fov_deg", str(float(args.fov_deg)),
               "--image_size", str(int(args.image_size)),
               "--top_ring_num", str(int(args.top_ring_num)),
               "--top_ring_elev_deg", str(float(args.top_ring_elev_deg)),
               "--top_ring_dist_scale", str(float(args.top_ring_dist_scale))]

        # ==== 可选参数与性能相关参数（按需追加） ====
        if getattr(args, "axis_correction", None):
            cmd += ["--axis_correction", str(args.axis_correction)]
        if getattr(args, "yaw_offset_deg", None):
            cmd += ["--yaw_offset_deg", str(float(args.yaw_offset_deg))]
        if getattr(args, "top_ring_start_azim_deg", None):
            cmd += ["--top_ring_start_azim_deg", str(float(args.top_ring_start_azim_deg))]
        if getattr(args, "seed", None) is not None:
            cmd += ["--seed", str(int(args.seed))]
        if getattr(args, "batch_chunk", None):
            cmd += ["--batch_chunk", str(int(args.batch_chunk))]
        if getattr(args, "bin_size", None) is not None:
            cmd += ["--bin_size", str(int(args.bin_size))]
        if getattr(args, "max_faces_per_bin", None) is not None:
            cmd += ["--max_faces_per_bin", str(int(args.max_faces_per_bin))]
        if getattr(args, "no_cull_backfaces", False):
            cmd += ["--no_cull_backfaces"]

        # 输出控制
        if getattr(args, "save_h5", False): cmd += ["--save_h5"]
        if getattr(args, "save_rgb_png", False): cmd += ["--save_rgb_png"]
        if getattr(args, "save_mask_png", False): cmd += ["--save_mask_png"]
        if getattr(args, "save_metric_depth", False): cmd += ["--save_metric_depth"]
        if getattr(args, "save_depth_png16", False): cmd += ["--save_depth_png16"]
        if getattr(args, "make_depth_video", False):
            cmd += ["--make_depth_video", "--depth_video_max_meters", str(int(args.depth_video_max_meters))]
        if getattr(args, "save_nocs", False): cmd += ["--save_nocs"]
        if getattr(args, "save_nocs_png8", False): cmd += ["--save_nocs_png8"]
        if getattr(args, "make_nocs_video", False): cmd += ["--make_nocs_video"]
        if getattr(args, "nocs_norm", None): cmd += ["--nocs_norm", str(args.nocs_norm)]
        if getattr(args, "nocs_equal_axis", False): cmd += ["--nocs_equal_axis"]
        if getattr(args, "check_nocs", False): cmd += ["--check_nocs"]
        if getattr(args, "make_video", False):
            cmd += ["--make_video", "--video_fps", str(int(args.video_fps))]
        if getattr(args, "input_format", None):
            cmd += ["--input_format", str(args.input_format)]
        if getattr(args, "use_uv_textures", False):
            cmd += ["--use_uv_textures"]
        if getattr(args, "obj_loader", None):
            cmd += ["--obj_loader", str(args.obj_loader)]
        if getattr(args, "atlas_size", None):
            cmd += ["--atlas_size", str(int(args.atlas_size))]
        if getattr(args, "use_atlas", False):
            cmd += ["--use_atlas"]
        if getattr(args, "atlas_mem_limit_gb", None):
            cmd += ["--atlas_mem_limit_gb", str(float(args.atlas_mem_limit_gb))]
        if getattr(args, "overwrite_render", False):
            cmd += ["--overwrite"]

        # 关键：安静执行（避免刷屏）——保留错误输出与本进程的关键信息
        rc = sh(cmd, dry=getattr(args, "dry_run", False), quiet=getattr(args, "quiet_child", False))
        dt = time.perf_counter() - t0
        print(f"[render/batch] done rc={rc}  total={total}  time={dt:.2f}s  avg={dt/max(total,1):.2f}s/obj")
        return
    
    print(f"[render/single] cat={category}  total={total}")
    t0 = time.perf_counter()
    iterator = enumerate(models_list, 1)
    if getattr(args, "progress", False) and tqdm is not None:
        iterator = zip(range(1, total+1), tqdm(models_list, total=total, ncols=80, desc=f"{category}"))

    for idx, model in iterator:
        model = Path(model)
        ext = model.suffix.lower()
        out_dir = render_out_dir_for_any(
            model, P.renders, args.render_scheme,
            dl_backend=getattr(args, "dl_backend", "custom"),
            oxl_out_cat=OXL_OUT_CAT
        )

        target_video = out_dir / "orbit_rgb.mp4" if args.make_video else None
        if (not args.overwrite_render) and target_video and target_video.exists():
            print(f"[SKIP] {model.name} → video exists: {target_video}")
            continue

        cmd = [sys.executable, str(render_script),
               "--obj", str(model),
               "--out", str(out_dir),
               "--num_cams", str(int(args.num_cams)),
               "--elev_deg", str(float(args.elev_deg)),
               "--fov_deg", str(float(args.fov_deg)),
               "--image_size", str(int(args.image_size)),
               "--top_ring_num", str(int(args.top_ring_num)),
               "--top_ring_elev_deg", str(float(args.top_ring_elev_deg)),
               "--top_ring_dist_scale", str(float(args.top_ring_dist_scale)),
               ]
      
        if args.axis_correction:
            cmd += ["--axis_correction", str(args.axis_correction)]
        if args.yaw_offset_deg:
            cmd += ["--yaw_offset_deg", str(float(args.yaw_offset_deg))]
        if args.top_ring_start_azim_deg:
            cmd += ["--top_ring_start_azim_deg", str(float(args.top_ring_start_azim_deg))]
        if args.seed is not None:
            cmd += ["--seed", str(int(args.seed))]
        if args.batch_chunk:
            cmd += ["--batch_chunk", str(int(args.batch_chunk))]
        if args.save_h5:
            cmd += ["--save_h5"]
        if category is not None:
            cmd += ["--label", str(category)]
        if args.save_rgb_png:
            cmd += ["--save_rgb_png"]
        if args.save_mask_png:
            cmd += ["--save_mask_png"]
        if args.save_metric_depth:
            cmd += ["--save_metric_depth"]
        if args.save_depth_png16:
            cmd += ["--save_depth_png16"]
        if args.make_depth_video:
            cmd += ["--make_depth_video", "--depth_video_max_meters", str(int(args.depth_video_max_meters))]
        if args.save_nocs:
            cmd += ["--save_nocs"]
        if args.save_nocs_png8:
            cmd += ["--save_nocs_png8"]
        if args.make_nocs_video:
            cmd += ["--make_nocs_video"]
        if args.nocs_norm:
            cmd += ["--nocs_norm", str(args.nocs_norm)]
        if args.nocs_equal_axis:
            cmd += ["--nocs_equal_axis"]
        if args.check_nocs:
            cmd += ["--check_nocs"]
        if args.make_video:
            cmd += ["--make_video", "--video_fps", str(int(args.video_fps))]
        if args.no_cull_backfaces:
            cmd += ["--no_cull_backfaces"]
        if args.input_format:
            cmd += ["--input_format", str(args.export_format)]
        # 贴图：GLB 可不需要，OBJ 强烈建议开启（即使用户未传）
        if args.use_uv_textures or ext == ".obj":
            cmd += ["--use_uv_textures"]

        # OBJ 加载器参数（你在 render_orbit_* 中已实现）
        if args.obj_loader:
            cmd += ["--obj_loader", str(args.obj_loader)]
        if args.atlas_size:
            cmd += ["--atlas_size", str(int(args.atlas_size))]
        if args.use_atlas:
            cmd += ["--use_atlas"]
        if args.atlas_mem_limit_gb:
            cmd += ["--atlas_mem_limit_gb", str(float(args.atlas_mem_limit_gb))]    
        if args.overwrite_render:
            cmd += ["--overwrite"]

        rc = sh(cmd, dry=args.dry_run)
        if rc != 0:
            print(f"[WARN] Rendering failed for {model}")


# ---------- Collect models for downstream according to download backend ----------
def collect_models_oxl(P: PipelinePaths, out_cat: str, labels: Optional[List[str]] = None) -> List[Path]:
    """Collect model paths from OXL layout: downloads/<out_cat>/(<label>/)?<sha>/raw/... (no _pool)."""
    base = P.downloads / out_cat
    candidates: List[Path] = []
    # If labels specified, look into label subfolders; otherwise, collect from all labels and root
    label_dirs: List[Path] = []
    if labels:
        for lab in labels:
            d = base / lab
            if d.is_dir():
                label_dirs.append(d)
    else:
        # all immediate subdirs
        for d in sorted([p for p in base.iterdir() if p.is_dir()]):
            label_dirs.append(d)
        # also include root level instances (no label)
        label_dirs.append(base)

    for lab_dir in label_dirs:
        # every sha under this label dir
        for sha_dir in sorted([p for p in lab_dir.iterdir() if p.is_dir() and p.name not in {"_pool"}]):
            raw = sha_dir / 'raw'
            if not raw.is_dir():
                continue
            # pick first mesh file
            m = _pick_mesh_file(raw)
            if m:
                candidates.append(m)
    return candidates

# --- add: 工具函数：把 YAML 写成 argparse 默认值 ---
def _apply_yaml_defaults_to_parser(parser: argparse.ArgumentParser, fp: Path):
    """
    从 YAML 读取配置并写入 parser.set_defaults(...)。
    支持两种结构：
      1) 扁平：{key: value}
      2) 分段：{"pipeline": {...}, "render": {...}}
    """
    import yaml  # 确保已安装 pyyaml
    with open(fp, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    if isinstance(cfg, dict) and ("pipeline" in cfg or "render" in cfg):
        for sec in ("pipeline", "render"):
            v = cfg.get(sec)
            if isinstance(v, dict):
                flat.update(v)
    elif isinstance(cfg, dict):
        flat.update(cfg)
    parser.set_defaults(**flat)

# --- add: 工具函数：在 YAML-only 模式下关闭所有 required ---
def _disable_required_options(parser: argparse.ArgumentParser):
    """
    把可选参数（带 -- 的那些）的 required=False，
    避免 parse_args([]) 仍因 required=True 报错。
    """
    for act in parser._actions:
        # 位置参数（如 act.option_strings == []）不动；只处理可选参数
        if getattr(act, "option_strings", None):
            try:
                act.required = False
            except Exception:
                pass

def _disable_required_options(parser: argparse.ArgumentParser):
    for act in parser._actions:
        if getattr(act, "option_strings", None):
            try:
                act.required = False
            except Exception:
                pass
# ---------- 3) 打印：最终配置 / 等效命令行 ----------
import argparse, json, sys, shlex, subprocess, copy
from pathlib import Path
from enum import Enum
def _normalize_steps(s):
    if s is None: return None
    if isinstance(s, str):
        parts = [t.strip() for t in s.replace(",", " ").split() if t.strip()]
        return parts or None
    if isinstance(s, (list, tuple, set)):
        return list(s)
    return s

def _maybe_split_commas(v):
    if isinstance(v, str):
        return [t.strip() for t in v.split(",") if t.strip()]
    return v

def _json_default(o):
    if isinstance(o, Path): return str(o)
    if isinstance(o, Enum): return o.value
    if isinstance(o, set):  return list(o)
    try:
        return str(o)
    except Exception:
        return repr(o)

def _sorted_vars(ns):
    d = {}
    for k, v in vars(ns).items():
        d[k] = str(v) if isinstance(v, Path) else v
    return {k: d[k] for k in sorted(d.keys())}

def _format_cmd(cmd):
    if isinstance(cmd, (list, tuple)):
        return shlex.join([str(x) for x in cmd])
    return str(cmd)

def _snapshot_factory_defaults() -> dict:
    # 获取“工厂默认值”（不带 YAML/CLI）
    p0 = build_argparser()
    _disable_required_options(p0)
    return vars(p0.parse_args([]))

def args_to_equivalent_cli(parser: argparse.ArgumentParser,
                           args: argparse.Namespace,
                           include_program: bool = True,
                           full: bool = False) -> str:
    """
    生成“等效命令行”。full=True 显示所有键；否则只显示“与工厂默认不同”的键。
    对于 store_true：仅 True 时输出；跳过所有 'no_*' 反向键。
    """
    import copy
    eff = copy.deepcopy(args)
    # eff = _resolve_negated_bools(eff)          
    if hasattr(eff, "steps"):
        eff.steps = _normalize_steps(eff.steps) 
    eff_dict = vars(eff)
    defaults = {} if full else _snapshot_factory_defaults()

    parts = []
    if include_program:
        parts = ["python", "mesh_pipeline_full.py"]

    for act in parser._actions:
        opts = getattr(act, "option_strings", None)
        if not opts:           # 位置参数跳过
            continue
        dest = getattr(act, "dest", None)
        if not dest or dest not in eff_dict:
            continue
        if dest.startswith("no_"):   # 跳过反向键
            continue

        val = eff_dict[dest]
        if (not full) and (dest in defaults) and defaults[dest] == val:
            continue

        # store_true: 仅 True 时打印一次 flag
        is_store_true = (
            getattr(act, "const", None) is True and
            getattr(act, "nargs", None) is None and
            getattr(act, "type", None) is None and
            isinstance(val, bool)
        )
        if is_store_true:
            if val: parts.append(opts[0])
            continue

        # 其他布尔：--key true/false
        if isinstance(val, bool):
            parts += [opts[0], str(val).lower()]
            continue

        # 列表
        if isinstance(val, (list, tuple)):
            if len(val) == 0: continue
            parts.append(opts[0]); parts += [str(x) for x in val]
            continue

        # None 跳过
        if val is None:
            continue

        # 标量
        parts += [opts[0], str(val)]

    return _format_cmd(parts)

# ---------- 4) 可选：把所有子进程命令在执行前打印出来 ----------
def wrap_subprocess_logging(args):
    if not getattr(args, "print_effective", False):
        return
    _orig_run = subprocess.run
    def _log_run(cmd, *a, **kw):
        print(">>> RUN:", _format_cmd(cmd)); sys.stdout.flush()
        return _orig_run(cmd, *a, **kw)
    subprocess.run = _log_run

    if hasattr(subprocess, "check_call"):
        _orig_check_call = subprocess.check_call
        def _log_check_call(cmd, *a, **kw):
            print(">>> RUN:", _format_cmd(cmd)); sys.stdout.flush()
            return _orig_check_call(cmd, *a, **kw)
        subprocess.check_call = _log_check_call

    if hasattr(subprocess, "call"):
        _orig_call = subprocess.call
        def _log_call(cmd, *a, **kw):
            print(">>> RUN:", _format_cmd(cmd)); sys.stdout.flush()
            return _orig_call(cmd, *a, **kw)
        subprocess.call = _log_call

    _orig_popen = subprocess.Popen
    class _LogPopen(subprocess.Popen):
        def __init__(self, cmd, *a, **kw):
            print(">>> RUN(POPEN):", _format_cmd(cmd)); sys.stdout.flush()
            super().__init__(cmd, *a, **kw)
    subprocess.Popen = _LogPopen

# ---------- CLI ----------
import os, contextlib, re

def _materialize_github_token(args) -> str | None:
    """
    解析 token 来源：优先 --github_token，其次 --github_token_file。
    支持：
      - 明文：ghp_xxx
      - env:VAR  → 从环境 VAR 读取；VAR 缺省用 args.github_token_env
      - file:/path/to/token → 从文件读取
    """
    tok = getattr(args, "github_token", None)
    if tok:
        if tok.startswith("env:"):
            var = tok.split(":", 1)[1] or getattr(args, "github_token_env", "GITHUB_TOKEN")
            return os.environ.get(var)
        if tok.startswith("file:"):
            path = tok.split(":", 1)[1]
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception:
                return None
        return tok  # 明文
    # 备选文件
    fpath = getattr(args, "github_token_file", None)
    if fpath:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    # 最后兜底直接读环境
    return os.environ.get(getattr(args, "github_token_env", "GITHUB_TOKEN"))

@contextlib.contextmanager
def github_auth_env(args):
    """
    在 with 块内注入 GITHUB_TOKEN / GH_TOKEN 与 GIT_TERMINAL_PROMPT=0。
    退出时恢复原值。
    """
    token = _materialize_github_token(args)
    old = {}
    keys = ["GITHUB_TOKEN", "GH_TOKEN", "GIT_TERMINAL_PROMPT"]
    for k in keys:
        old[k] = os.environ.get(k)

    if token:
        os.environ["GITHUB_TOKEN"] = token
        os.environ["GH_TOKEN"] = token  # 兼容两种变量名
    if getattr(args, "git_no_prompt", False) or True:
        os.environ["GIT_TERMINAL_PROMPT"] = "0"

    try:
        # 可选：打印一条脱敏提示
        if getattr(args, "print_effective", False):
            shown = "***" if token else None
            print(f"[auth] GitHub token set: {bool(token)}  (masked={shown is not None}), no_prompt={os.environ['GIT_TERMINAL_PROMPT']}")
        yield
    finally:
        # 恢复
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _mask_github_token_in_text(s: str) -> str:
    # 屏蔽 https://TOKEN@github.com
    return re.sub(r"https://([^/@:\s]+)@github\.com", "https://***@github.com", s)

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Mesh workflow pipeline")
    # --- NEW: pipeline YAML config ---
    p.add_argument("--config", type=Path, default=None, help="YAML config for this pipeline (CLI overrides YAML).")
    p.add_argument("--allow_cli_override", action=argparse.BooleanOptionalAction, default=False,
               help="当提供 --config 时，允许命令行参数覆盖 YAML。默认关闭：仅以 YAML 为准。")
    # Core paths
    p.add_argument("--root",  type=Path, help="Pipeline root directory (will create subfolders)") # disable the required true when using yaml
    p.add_argument("--png_root", type=Path, help="PNG root used by build_metadata.py to infer categories/shape_keys")
    p.add_argument("--maps_json", type=Path, default=None)
    p.add_argument("--maps_dir", type=Path, default=None, help="Directory containing mapping JSON files to merge (recursive).")
    p.add_argument("--maps_glob", type=str, default="*.json", help="Glob to select mapping JSONs inside --maps_dir (used with rglob).")
    p.add_argument("--dup_policy", choices=["prefer_last","prefer_first","error"], default="prefer_last", help="How to resolve duplicate keys when merging map files.")

    p.add_argument("--rot_json", type=Path, default=None, help="Fallback rotation JSON used when a per-category file is not found.")
    p.add_argument("--rot_dir", type=Path, default=None, help="Directory containing per-category rotation JSONs, structured as <rot_dir>/<category>/<rot_filename>.")
    p.add_argument("--rot_filename", type=str, default="final_shape_rotations.json", help="File name of per-category rotation JSON inside each category folder under --rot_dir.")

    # Optional custom download directory (maps to build_metadata.py --download_root)
    p.add_argument("--download_root", type=Path, default=None, help="If set, 3D assets will be downloaded under this directory, e.g., /data/<category>/<subA>/<subB>/...")
    p.add_argument("--overwrite_download", action=argparse.BooleanOptionalAction, default=False,) #action="store_true"
    # Binaries / scripts
    p.add_argument("--blender", type=str, help="Path to Blender executable (e.g., /snap/blender/current/blender)")
    p.add_argument("--build_script", default=Path(__file__).parent / "build_metadata.py", type=Path)
    p.add_argument("--align_script", default=Path(__file__).parent / "align_export_snapshot_blender.py", type=Path)
    p.add_argument("--render_script", default=Path(__file__).parent / "render_orbit_rgbd.py", type=Path)

    # Selection
    p.add_argument("--categories", nargs="*", default=None, help="Limit to these categories (default: let build_metadata infer)")
    p.add_argument("--ids_json", type=Path, default=None, help="Optional: JSON listing shape ids/keys to restrict")
    p.add_argument("--ids_txt", type=Path, default=None, help="Optional: TXT of shape_keys, one per line")

    # Download step opts
    p.add_argument("--debug_limit", type=int, default=None)

    # Download backend selection
    p.add_argument("--dl_backend", choices=["custom","oxl"], default="custom",
                   help="Which downloader to use: custom(build_metadata.py) or oxl(ObjaverseXL)")
    # OXL options
    p.add_argument("--oxl_source", type=str, default="sketchfab", choices=["sketchfab","github"],
                   help="ObjaverseXL source to use")
    p.add_argument("--label_json", type=Path, default=None, help="Path to label json for OXL filtering")
    p.add_argument("--labels", type=str, default=None, help="Comma-separated labels to filter in OXL")
    p.add_argument("--label_match_mode", type=str, default="any", choices=["any","all"],
                   help="Label match mode for OXL filtering")
    # Per-category OXL options
    p.add_argument("--label_dir", type=Path, default=None,
                   help="Directory containing per-category label JSONs for per_category OXL mode (can reuse --rot_dir)")
    p.add_argument("--sha_index_csv", type=Path, default=None,
                   help="Path to SHA index CSV containing all sha256 mappings for per_category OXL mode")
    # p.add_argument("--oxl_per_category_mode", action="store_true",
    #                help="Enable per_category OXL download mode (requires --label_dir and --sha_index_csv)")
    p.add_argument("--oxl_out_category", type=str, default="ObjaverseXL_github", help="Category name used to store OXL downloads")
    p.add_argument("--oxl_labels", nargs="*", default=None, help="Only use these labels (subfolders) for OXL downstream render")
    p.add_argument("--oxl_stage_dir", type=Path, default=None, help="Staging directory for raw downloads (zips) before consolidation; default: <download_root>/<oxl_out_category>/_stage")
    p.add_argument("--oxl_keep_stage", action=argparse.BooleanOptionalAction, default=False, help="Keep files in staging directory after consolidation (default: remove when possible)")
    p.add_argument("--oxl_keep_pool", action=argparse.BooleanOptionalAction, default=False, help="Keep <out_cat>/_pool after consolidation (default: remove).")
    p.add_argument("--oxl_prune_zips", action=argparse.BooleanOptionalAction, default=False, help="After successful extraction and mesh pick, delete original archive to save space")
    p.add_argument("--oxl_prefer_label_root", action=argparse.BooleanOptionalAction, default=False, help="When consolidating, prefer to place raw files under <out_cat>/<label>/<sha>/raw instead of <out_cat>/<sha>/raw")
    p.add_argument("--oxl_prune_mesh_dir", action=argparse.BooleanOptionalAction, default=False,
                    help="After assets are completed into <sha>/raw, remove the temporary <sha>/mesh directory.")
    p.add_argument("--overwrite_download_repo", action=argparse.BooleanOptionalAction, default=False,
                   help="强制重下仓库/zip：会清空 OXL 的 _stage 并重新下载。")
    p.add_argument("--overwrite_consolidate", action=argparse.BooleanOptionalAction, default=False,
                   help="强制重做 consolidate：清理 <out_cat>/<label>/<sha> 后重建，即使不重下 zip。")
    p.add_argument("--oxl_force_reconsolidate", action=argparse.BooleanOptionalAction, default=False,
                   help="当 OXL 没有返回新条目(df为空)时，仍然从 _stage 扫描现有 zip 进行归拢/确保资产。")
    p.add_argument("--oxl_refresh_metadata", action=argparse.BooleanOptionalAction, default=False,
               help="Force re-fetch OXL metadata even if metadata.csv exists.")
    p.add_argument("--oxl_consolidate_link_mode", type=str, default="hardlink", choices=["hardlink", "copy", "symlink"],
               help="How to materialize files into <label>/<sha>/raw (default: hardlink; cross-device falls back to copy).")
    # Align step opts
    p.add_argument("align_out_root", nargs="?", type=Path, default=None,
                   help="Optional root directory for aligned models (default: <download_root>/<out_cat>")
    p.add_argument("--normalize_unit", action=argparse.BooleanOptionalAction, default=False) 
    p.add_argument("--target_size", type=float, default=1.0)
    p.add_argument("--grid_n", type=int, default=None)
    p.add_argument("--prefer_gpu", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--topdown", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--clean_aligned_dir", action=argparse.BooleanOptionalAction, default=False, help="Remove existing GLBs before align-export")
    p.add_argument("--limit", type=int, default=None, help="Only process first N shapes in Blender step")
    p.add_argument("--require_hollow_for", nargs="*", default=None, help="Which categories require hollow/open checks")
    p.add_argument("--overwrite_align", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--export_format", choices=["glb","obj"], default="obj", help="Export format for aligned models (default: glb).")
    # Render step opts
    p.add_argument("--render_scheme", choices=["flat","hier"], default="flat")
    p.add_argument("--render_after_each_category", action=argparse.BooleanOptionalAction, default=False, help="Interleave stage 2+3 per category: align then immediately render that category before moving on.")
    
    # Step control
    p.add_argument("--steps", nargs="+", choices=["download", "align", "render"], default=["download", "align", "render"],
                   help="Which pipeline steps to run, in order. Any subset of: download align render.")
    # Per-category execution (rot_dir-driven)
    p.add_argument("--per_category_flow", action=argparse.BooleanOptionalAction, default=False,
                   help="Iterate categories from --rot_dir (or --categories if given) and run steps per category: download -> align -> render.")
    p.add_argument("--rot_merge_glob", type=str, default="*_cluster_*_rotations.json", help="当 rot_dir/<cat>/<rot_filename> 不存在时，匹配并合并这些文件。")
    p.add_argument("--rot_dup_policy", choices=["prefer_last","prefer_first","error"], default="prefer_last", help="合并 cluster json 时重复 shape id 的处理策略。")
    p.add_argument("--persist_merged_rot", action=argparse.BooleanOptionalAction, default=False, help="合并后是否持久化写入 rot_dir/<cat>/<rot_filename>（默认写回）。")
    p.add_argument("--force_remerge", action=argparse.BooleanOptionalAction, default=False, help="即便已存在 rot_filename 也强制重新根据 cluster 文件合并一份。")
   
    # Render step opts
    p.add_argument("--num_cams", type=int, default=80)
    p.add_argument("--elev_deg", type=float, default=20.0)
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--top_ring_num", type=int, default=40)
    p.add_argument("--top_ring_elev_deg", type=float, default=78)
    p.add_argument("--top_ring_dist_scale", type=float, default=0.85)
    p.add_argument("--use_uv_textures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no_cull_backfaces", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--axis_correction', type=str, default='none', choices=['none','y_up_to_z_up','z_up_to_y_up'])
    p.add_argument("--seed", type=int, default=123, help="Random seed for rendering (default: None)")
    p.add_argument("--batch_chunk", type=int, default=0)
    p.add_argument("--save_rgb_png", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_mask_png", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_metric_depth", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_depth_png16", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--make_depth_video', action=argparse.BooleanOptionalAction, default=False, help='把深度序列导出为灰度 mp4（可视化）')
    p.add_argument('--depth_video_max_meters', type=float, default=0.0, help='深度视频上限米数；<=0 则用 99th 百分位自动设定')
    p.add_argument("--make_video", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--video_fps", type=int, default=24)
    p.add_argument('--yaw_offset_deg', type=float, default=0.0,
                help='全局偏航（度）。对所有方位角统一加偏移，用于补偿坐标/导出轴向变化（例：180）。')
    p.add_argument('--top_ring_start_azim_deg', type=float, default=None,
                help='顶部环起始方位角；默认与 equatorial 的起始方位（+yaw_offset）保持一致。')
    # ---h5py ---
    p.add_argument('--save_h5', action=argparse.BooleanOptionalAction, default=False,)
    # --- NOCS 输出相关 ---
    p.add_argument('--save_nocs', action=argparse.BooleanOptionalAction, default=False,
                help='保存 NOCS 的 float32 .npy（H,W,3，范围[0,1]）')
    p.add_argument('--save_nocs_png8', action=argparse.BooleanOptionalAction, default=False,
                help='保存 NOCS 的 8-bit PNG 可视化（RGB=xyz∈[0,255]）')
    p.add_argument('--make_nocs_video', action=argparse.BooleanOptionalAction, default=False,
                help='把 NOCS 序列导出为 mp4（RGB=xyz∈[0,255]）')
    p.add_argument('--nocs_norm', choices=['bbox','center_scale'], default='bbox',
                help='NOCS 归一化方式：bbox 线性到 [0,1]^3；或 center_scale 等比缩放到 [-0.5,0.5]^3 再平移到 [0,1]^3')
    p.add_argument('--nocs_equal_axis', action=argparse.BooleanOptionalAction, default=False,
                help='与 center_scale 联用：用最长边作为统一尺度（等比）；否则对每轴分别缩放（与 bbox 等价）')
    p.add_argument('--check_nocs', action=argparse.BooleanOptionalAction, default=False,
              help='在渲染过程中检查 NOCS 跨帧颜色一致性并打印统计')
    p.add_argument('--nocs_check_stride', type=int, default=8,
                help='抽样步长（像素网格步进）；越大越快、越粗略')
    p.add_argument('--nocs_check_voxel', type=float, default=0.02,
                help='参考 NOCS 的体素量化步长，用于跨帧对齐同一物体点（0~1 归一化域）')
    p.add_argument('--nocs_check_topk', type=int, default=8,
                help='打印逐帧误差最大的前 K 帧')
    
    p.add_argument("--overwrite_render", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--obj_loader', type=str, default='auto', choices=['auto','trimesh','p3d'],
               help="OBJ 用哪种加载器：auto(默认)/trimesh/p3d(推荐，多材质稳)")
    p.add_argument('--atlas_size', type=int, default=512,
               help="当使用 p3d 加载 OBJ 时，创建纹理图集的分辨率（每面采样大小）")
    p.add_argument('--use_atlas', action=argparse.BooleanOptionalAction, default=False,
               help='为 OBJ 创建 per-face 纹理图集（非常吃内存）。默认关闭，使用 UV 贴图。')
    p.add_argument('--atlas_mem_limit_gb', type=float, default=2.0,
               help='估算 atlas 内存超过该上限则自动改用 UV。默认 2GB。')
    p.add_argument('--input_format', type=str, choices=['glb','obj'], default=None,
               help="输入模型格式，默认自动识别。若你确认全是某种格式可指定以加速。")
    p.add_argument('--bin_size', type=int, default=64, help="渲染时按模型复杂度分批，每批该值个模型（越小越省内存但越慢）。默认100。")
    p.add_argument('--max_faces_per_bin', type=int, default=200000, help="渲染时按模型复杂度分批，每批最大面数（越小越省内存但越慢）。默认200000。")
    # General
    p.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    # --- NEW: batch render mode & extra ---
    p.add_argument("--render_batch_mode", choices=["batch", "single"], default="batch",
                   help="batch: 单次调用渲染脚本处理该 category 的清单；single: 逐个模型调用（兼容旧法，便于统计单模型耗时）")
    p.add_argument("--print_effective", action=argparse.BooleanOptionalAction, default=False,
               help="打印最终生效的参数（完整字典）与组装的伪命令行，便于检查。")
    p.add_argument("--github_token",
               type=str, default=None,
               help="GitHub token (PAT). 支持直接写明文，或以 'env:VAR' 读取环境变量，或 'file:/path/to/token' 读取文件。")
    p.add_argument("--git_no_prompt",
               action=argparse.BooleanOptionalAction, default=False,
               help="禁用 git 交互式密码提示（GIT_TERMINAL_PROMPT=0）。默认建议开启。")   
    return p


def main():
    parser = build_argparser()
    # 预解析只拿到 --config / --allow_cli_override
    pre_args, _ = parser.parse_known_args()

    use_yaml = getattr(pre_args, "config", None) is not None
    allow_cli = bool(getattr(pre_args, "allow_cli_override", False))

    if use_yaml:
        # 先把 YAML 写进默认值池
        _apply_yaml_defaults_to_parser(parser, pre_args.config)

    if use_yaml and not allow_cli:
        # YAML-only 模式：关闭 required，忽略 CLI（除了 --config/--allow_cli_override）
        _disable_required_options(parser)
        args = parser.parse_args([])   # 只使用 YAML + argparse 默认值
    else:
        # 普通模式或允许 CLI 覆盖：正常解析（CLI 可覆盖 YAML）
        args = parser.parse_args()

    if hasattr(args, "categories"):
        args.categories = _maybe_split_commas(args.categories)
    if hasattr(args, "oxl_labels"):
        args.oxl_labels = _maybe_split_commas(args.oxl_labels)
    if hasattr(args, "labels"):
        args.labels = _maybe_split_commas(args.labels)
    if hasattr(args, "steps"):
        args.steps = _maybe_split_commas(args.steps)
    
    # if allow_cli and args.print_effective:
    #     wrap_subprocess_logging(args) # print subprocess commands if --print_effective
      
    if args.print_effective:
        import copy
        eff = copy.deepcopy(args)
        if hasattr(eff, "steps"):
            eff.steps = _normalize_steps(eff.steps) # 若你没定义这个函数，也可先注释掉
        # print("\n=== [EFFECTIVE CONFIG] ===")
        # print(json.dumps(_sorted_vars(eff), indent=2, ensure_ascii=False, default=_json_default))
        # print("\n=== [EQUIVALENT CLI | DIFF] ===")
        print(args_to_equivalent_cli(parser, eff, include_program=True, full=False))
        print("\n=== [EQUIVALENT CLI | FULL] ===")
        print(args_to_equivalent_cli(parser, eff, include_program=True, full=True))

    args.download_root = (args.root / "G-objaverse") if args.download_root is None else args.download_root.resolve()
    dl = args.download_root.resolve() if (args.download_root or args.dl_backend == "oxl") else (args.root / "G-objaverse") 
    args.png_root = args.png_root.resolve() if args.png_root else (args.root / "G-objaverse")
    P = PipelinePaths.make(args.root.resolve(), dl, args.dl_backend)

    steps = [s.lower() for s in args.steps]
    steps_set = set(steps)
    

    # === Per-category flow: download -> align -> render for each cat ===
    if args.per_category_flow:
        # Determine categories
        if args.categories and args.dl_backend != "oxl":
            cats = _norm_label_list(args.categories)
        elif args.oxl_labels and args.dl_backend == "oxl":
            cats = _norm_label_list(args.oxl_labels)
        else:
            if not getattr(args, "rot_dir", None):
                raise ValueError("--per_category_flow requires --rot_dir or explicit --categories")
            cats = infer_categories_from_rot_dir(Path(args.rot_dir), args.rot_filename)
            if not cats:
                raise FileNotFoundError(f"No categories found under --rot_dir={args.rot_dir} containing {args.rot_filename}")
            
        print(f"=== Per-category flow over categories: {cats} ===")
        for cat in cats:
            print(f"===== CATEGORY: {cat} =====")
            # Step 1: download (restricted to this category)
            if "download" in steps_set and getattr(args, "dl_backend", "custom") == "oxl":
                # 检查必需的参数
                if not getattr(args, 'label_dir', None) and not getattr(args, 'rot_dir', None):
                    raise ValueError("per_category OXL mode requires --label_dir (or --rot_dir)")
                if not getattr(args, 'sha_index_csv', None):
                    raise ValueError("per_category OXL mode requires --sha_index_csv")
                step_download_assets_oxl_per_category(args, P, cat)
                # else:
                #     step_download_assets_oxl(args, P)
            elif "download" in steps_set and args.dl_backend == "custom":
                print("--- [cat] Step 1: Download / verify assets ---")
                step_download_assets(args, P, inferred_cats=[cat], cat=[cat])
            else:
                print("[SKIP] [cat] download")

            # Step 2: align for this category
            align_out_dir = None
            if "align" in steps_set:
                print("--- [cat] Step 2: Align & export GLBs ---")
                align_out_dir = step_align_export_one_category(args, P, cat)
            else:
                print("[SKIP] [cat] align & export")
                # If we didn't align now, but will render, try to locate existing dir
                if "render" in steps_set:
                    maybe = P.aligned_glb / cat
                    if args.dl_backend == "oxl" and args.align_out_root:
                        maybe = Path(args.align_out_root) / cat
                    elif args.dl_backend == "oxl" and not args.align_out_root:
                        maybe = Path(args.download_root) / OXL_OUT_CAT / cat
                    align_out_dir = maybe if maybe.is_dir() else None

            # Step 3: render for this category
            if "render" in steps_set:
                print("--- [cat] Step 3: Render orbit RGBD & videos ---")
                if align_out_dir is None:
                    maybe = P.aligned_glb / cat
                    if args.dl_backend == "oxl" and args.align_out_root:
                        maybe = Path(args.align_out_root) / cat
                    elif args.dl_backend == "oxl" and not args.align_out_root:
                        maybe = Path(args.download_root) / OXL_OUT_CAT / cat
                    if not maybe.is_dir():
                        print(f"[WARN] No aligned directory for {cat}; skipping render.")
                        continue
                    align_out_dir = maybe
                glb_list = list(iter_glbs(align_out_dir)) # collected corresponding type data as list passing 
                obj_list = list(iter_objs(align_out_dir))
                models = glb_list if args.input_format == "glb" else (obj_list if args.input_format == "obj" else (glb_list if glb_list else obj_list))
                print(f"[INFO] {cat}: to render → {len(models)} item(s) ({'GLB' if models==glb_list else 'OBJ'}) → {args.render_scheme} storage")
                step_render(args, P, models, cat)

            else:
                print("[SKIP] [cat] render")
        print("[ALL DONE] Per-category flow outputs:")
        print("- Downloads:", P.downloads)
        print("- Aligned GLBs:", P.aligned_glb)
        print("- Renders:", P.renders)
        return

    # === Step 1: Download / verify assets ===
    # Categories may be needed for ids extraction; infer early when using rot_dir
    cats_for_infer: List[str] = []
    if ("download" in steps_set or "align" in steps_set) and getattr(args, "rot_dir", None):
        cats_for_infer = infer_categories_from_rot_dir(Path(args.rot_dir), args.rot_filename)

    if "download" in steps_set:
        print("=== Step 1: Download / verify assets ===")
        step_download_assets(args, P, inferred_cats=cats_for_infer if cats_for_infer else None)
    else:
        print("[SKIP] Step 1: download")

    # === Category/selection resolution depending on which steps will run ===
    cats: List[str] = args.categories or []
    if not cats:
        if "align" in steps_set or "download" in steps_set:
            if getattr(args, "rot_dir", None):
                cats = infer_categories_from_rot_dir(Path(args.rot_dir), args.rot_filename)
                if not cats:
                    raise FileNotFoundError(f"No categories found under --rot_dir={args.rot_dir} containing {args.rot_filename}")
            else:
                blacklist = {"downloads", "aligned_glb", "renders", "logs", "manifests", "__pycache__", ".git", ".idea"}
                cats = [
                    d.name for d in sorted(Path(args.png_root).iterdir())
                    if d.is_dir() and d.name not in blacklist
                ]
        elif "render" in steps_set:
            cats = [d.name for d in sorted(P.aligned_glb.iterdir()) if d.is_dir()]

    aligned_dirs: List[Path] = []

    # === Step 2+3: Interleaved per category ===
    if args.render_after_each_category and ("align" in steps_set) and ("render" in steps_set):
        print(f"=== Step 2+3 (interleaved): categories: {cats} ===")
        for cat in cats:
            out_dir = step_align_export_one_category(args, P, cat)
            print(f"=== Step 3 (interleaved): Render {cat} ===")
            glb_list = list(iter_glbs(out_dir))
            obj_list = list(iter_objs(out_dir))
            models = glb_list if args.input_format == "glb" else (obj_list if args.input_format == "obj" else (glb_list if glb_list else obj_list))
            print(f"[INFO] {cat}: to render → {len(models)} item(s) ({'GLB' if models==glb_list else 'OBJ'}) → {args.render_scheme} storage")
            step_render(args, P, models, cat)
        print("[ALL DONE] Outputs:")
        print("- Downloads:", P.downloads)
        print("- Aligned GLBs:", P.aligned_glb)
        print("- Renders:", P.renders)
        return

    # === Step 2: Align & export GLBs ===
    if "align" in steps_set:
        print(f"=== Step 2: Align & export GLBs (categories: {cats}) ===")
        for cat in cats:
            out_dir = step_align_export_one_category(args, P, cat)
            aligned_dirs.append(out_dir)
    else:
        print("[SKIP] Step 2: align & export")

    # === Step 3: Render ===
    if "render" in steps_set:
        print("=== Step 3: Render orbit RGBD & videos ===")
        if getattr(args, "dl_backend", "custom") == "oxl":
            out_cat = getattr(args, 'oxl_out_category', 'oxl')
            models = collect_models_oxl(P, out_cat, labels=getattr(args, 'oxl_labels', None))
            print(f"[INFO] OXL downstream: render {len(models)} item(s) from labels={getattr(args,'oxl_labels',None)}")
            step_render(args, P, models, cat)
        else:
            if not aligned_dirs:
                if cats:
                    aligned_dirs = [P.aligned_glb / c for c in cats if (P.aligned_glb / c).is_dir()]
                else:
                    aligned_dirs = [d for d in sorted(P.aligned_glb.iterdir()) if d.is_dir()]
            for cat_dir in aligned_dirs:
                glb_list = list(iter_glbs(cat_dir))
                obj_list = list(iter_objs(cat_dir))
                models = glb_list if args.input_format == "glb" else (obj_list if args.input_format == "obj" else (glb_list if glb_list else obj_list))
                print(f"[INFO] {cat}: to render → {len(models)} item(s) ({'GLB' if models==glb_list else 'OBJ'}) → {args.render_scheme} storage")
                step_render(args, P, models, cat)
    else:
        print("[SKIP] Step 3: render")

    print("[ALL DONE] Outputs:")
    print("- Downloads:", P.downloads)
    print("- Aligned GLBs:", P.aligned_glb)
    print("- Renders:", P.renders)


if __name__ == "__main__":
    main()
