#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照已下载 PNG 的目录结构（/category/subA/subB/...）解析出 (category, shape_key)，
再从“总 JSON”中索引下载对应 3D 资源，解压、找网格文件，并建立/更新 metadata.csv。

示例使用：
python index_and_download_meshes.py \
  --png_root /Simulation-pipeline/Data/G-objaverse \
  --maps_json /path/to/all_categories_id_to_urls.json \
  --categories mug \
  --debug_limit 20

若你目前只有单类映射（例如 /mnt/data/mug_id_to_urls.json），也可以直接：
python index_and_download_meshes.py \
  --png_root /Simulation-pipeline/Data/G-objaverse \
  --maps_json /mnt/data/mug_id_to_urls.json \
  --categories mug
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
from urllib.parse import urlparse, urlsplit, urlunsplit, urljoin, quote
import subprocess
import sys
import glob 

# 放在文件顶部 imports 附近
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# from trellis import ObjaverseXL
# FULL REPO SEARCH：0=就近搜索(默认)；1=全仓库递归
FULL_REPO_SEARCH = os.environ.get("FBX_FULL_REPO_SEARCH", "0") == "1"
FBX_LOG_ORIGIN = os.environ.get("FBX_LOG_ORIGIN", "0") == "1"

# 贴图输出目录：fbm=放 <fbx>.fbm/（默认）；adjacent=放到 fbx 同目录
TEX_OUT_MODE = os.environ.get("FBX_TEX_OUT", "fbm")  # "fbm" 或 "adjacent"
USE_GITHUB_API_FALLBACK = os.environ.get("FBX_USE_GITHUB_API_FALLBACK", "1") == "1"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
_GITHUB_TREE_CACHE = {}

MESH_EXT_PRIORITY = [".blend", ".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply"]
ARCHIVE_EXT = [".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz"]
DL_WORKERS = 16  # 默认并行下载线程数

# ---- persistent cache for github tree (minimal) ----
import json, atexit
_GITHUB_TREE_CACHE_FILE = None  # set later after we know raw_dir
_GITHUB_TREE_CACHE = {} if '_GITHUB_TREE_CACHE' not in globals() else _GITHUB_TREE_CACHE

def _git_tree_cache_load(fp):
    global _GITHUB_TREE_CACHE
    try:
        p = str(fp)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # keys are tuples -> store as "user|repo|sha"
        if isinstance(data, dict):
            _GITHUB_TREE_CACHE.update({tuple(k.split("|", 2)): v for k, v in data.items()})
    except Exception:
        pass

def _git_tree_cache_save(fp):
    try:
        p = str(fp); d = {}
        for k, v in _GITHUB_TREE_CACHE.items():
            if isinstance(k, tuple):
                d["|".join(k)] = v
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:
        pass

_IMG_EXTS = (".png",".jpg",".jpeg",".tga",".tif",".tiff",".bmp",".exr",".dds",".ktx",".ktx2",".webp")

def _endswith_parts(p: Path, tail: Path) -> bool:
    """是否 p 的路径以 tail 的各段为后缀（大小写不敏感，分隔符兼容）。"""
    ps = [x.lower() for x in p.as_posix().split('/')]
    ts = [x.lower() for x in tail.as_posix().split('/')]
    if len(ts) > len(ps): 
        return False
    return ps[-len(ts):] == ts

_EQUIV_EXT_GROUPS = [
    {".jpg", ".jpeg", ".png"},          # 常见漫反射贴图
]
def _ext_variants(ext: str) -> set[str]:
    ext = ext.lower()
    for g in _EQUIV_EXT_GROUPS:
        if ext in g:
            return g
    return {ext}

def _endswith_parts(p: Path, tail: Path) -> bool:
    ps = [x.lower() for x in p.as_posix().split('/')]
    ts = [x.lower() for x in tail.as_posix().split('/')]
    if len(ts) > len(ps):
        return False
    return ps[-len(ts):] == ts

def _local_find_one(name: Path, roots: list[Path]) -> Path | None:
    # 规范化输入（别把 C:\ 当相对路径）
    rel = Path(_normalize_relpath(name.as_posix() if hasattr(name, "as_posix") else str(name)))
    bn  = rel.name

    # 1) 相对路径直拼（命中概率最高）
    for r in roots:
        cand = (r / rel)
        if cand.exists() and cand.is_file():
            return cand

    # 2) basename 精确查找（不做扩展名推断）
    for r in roots:
        hits = [h for h in r.rglob(bn) if h.is_file()]
        if hits:
            return hits[0]

    # 3) ★全局后缀匹配（不改扩展名、不猜测）：先用 2 段尾部，再用 1 段
    tails = []
    parts = rel.parts
    if len(parts) >= 2:
        tails.append(Path("/".join(parts[-2:])))
    tails.append(Path(bn))  # 兜底

    for r in roots:
        for h in r.rglob("*"):
            try:
                if h.is_file() and any(_endswith_parts(h, t) for t in tails):
                    return h
            except Exception:
                continue
    # 4) ★ 等价扩展名匹配（只在前面都失败时触发）
    stem = Path(bn).stem
    ext  = Path(bn).suffix.lower()
    variants = _ext_variants(ext)
    # 优先使用与原扩展“同组但不同后缀”的候选，避免无关命中
    alt_exts = [e for e in variants if e != ext]
    if alt_exts:
        # 先按 “同 stem + 等价后缀” 精确匹配
        for r in roots:
            for e in alt_exts:
                cand = stem + e
                hits = [h for h in r.rglob(cand) if h.is_file()]
                if hits:
                    return hits[0]
        # 再不区分大小写
        stem_low = stem.lower()
        for r in roots:
            hits = [h for h in r.rglob("*")
                    if h.is_file()
                    and h.stem.lower() == stem_low
                    and h.suffix.lower() in alt_exts]
            if hits:
                return hits[0]

    return None
# ----------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_json", type=str, default=None,
                    help="canonicalization JSON，含 shape_id 列表（可在 'shape_rotations' 下）")
    ap.add_argument("--ids_txt", type=str, default=None,
                    help="每行一个 shape_key（即 'subA/subB'）")
    ap.add_argument("--png_root", required=True, type=str,
                    help="已下载 PNG 的数据根目录，例如 /Simulation-pipeline/Data/G-objaverse")
    ap.add_argument("--maps_json", required=True, type=str,
                    help="总 JSON（或单类片段 JSON）文件路径")
    ap.add_argument("--categories", nargs="+", default=None,
                    help="只处理这些类别（默认自动从 png_root 下的一级目录推断）")
    ap.add_argument("--download_root", type=str, default=None,
                    help="3D 资源下载与解压的根目录（默认与 png_root 相同）")
    ap.add_argument("--metadata_csv", type=str, default=None,
                    help="metadata.csv 输出路径（默认在 download_root/metadata.csv）")
    ap.add_argument("--debug_limit", type=int, default=None,
                    help="仅处理前 N 个实例用于快速测试")
    ap.add_argument("--skip_extract", action="store_true",
                    help="只下载不解压（某些远端可能直接是可渲染格式）")
    ap.add_argument("--overwrite", action="store_true",
                    help="已存在文件也重新下载/解压")
    ap.add_argument("--report_csv", type=str, default=None,
                help="下载全量报告（含失败原因）；默认与 metadata.csv 同目录下的 download_report.csv")
    ap.add_argument('--dl_workers', type=int, default=16, help='Number of parallel download workers')
    ap.add_argument('--xl_download', action='store_true', help='Download using Trellis pipeline')
    return ap.parse_args()

def load_maps(maps_json_path: str, categories: Optional[List[str]]) -> Dict[str, Dict[str, str]]:
    """
    返回标准化映射：{category: {shape_key: url, ...}, ...}
    兼容：
      1) {"mug": {"1113/5569565": "http..."} , "chair": {...}}
      2) "mug": {"1113/5569565": "http..."} （不带最外层花括号的片段）
    """
    text = Path(maps_json_path).read_text(encoding="utf-8").strip()
    # 尝试直接解析成 JSON
    try:
        raw = json.loads(text)
        if isinstance(raw, dict):
            # 如果像 {"mug": {...}} 直接返回
            if all(isinstance(v, dict) for v in raw.values()):
                return {k: {kk: vv for kk, vv in v.items()} for k, v in raw.items()}
            # 如果已经是 {shape_key: url}
            if all(isinstance(v, str) for v in raw.values()):
                # 需要知道类别名；如果传入 categories 且只有 1 项，则归到该类
                if categories and len(categories) == 1:
                    return {categories[0]: raw}
                else:
                    raise ValueError("提供的 maps_json 是单类结构，但未通过 --categories 唯一指定该类名。")
        raise ValueError("maps_json 结构不符合预期。")
    except json.JSONDecodeError:
        # 可能是 '"mug": { ... }' 这类片段；补个大括号再 parse
        wrapped = "{\n" + text + "\n}"
        raw = json.loads(wrapped)
        # 只会有单个键，如 "mug"
        if len(raw) != 1:
            raise ValueError("片段 JSON 格式异常：顶层键不唯一。")
        k = next(iter(raw.keys()))
        if not isinstance(raw[k], dict):
            raise ValueError("片段 JSON 顶层值不是字典。")
        return {k: raw[k]}

def guess_categories_from_png_root(png_root: Path) -> List[str]:
    return sorted([p.name for p in png_root.iterdir() if p.is_dir() and not p.name.startswith(".")])

def iter_instances_from_png(png_root: Path, categories: List[str]) -> Iterable[Tuple[str, str, Path]]:
    """
    从 PNG 目录结构推断实例：
      期望形如：png_root / <category> / <subA> / <subB> / ...（里面包含 png）
      返回 (category, shape_key="subA/subB", instance_dir=<path to subB目录>)
    """
    for cat in categories:
        cat_dir = png_root / cat
        if not cat_dir.is_dir():
            continue
        # subA 层
        for subA in sorted([d for d in cat_dir.iterdir() if d.is_dir()]):
            for subB in sorted([d for d in subA.iterdir() if d.is_dir()]):
                shape_key = f"{subA.name}/{subB.name}"
                yield (cat, shape_key, subB)

def load_id_set(ids_json: Optional[str], ids_txt: Optional[str]) -> List[str]:
    S = set()
    if ids_json:
        J = json.loads(Path(ids_json).read_text(encoding="utf-8"))
        rot = J.get("shape_rotations", J)
        for k in rot.keys():
            if isinstance(k, str) and "/" in k:
                S.add(k)
    if ids_txt:
        for line in Path(ids_txt).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "/" in line:
                S.add(line)
    return sorted(S)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def is_archive(fname: str) -> bool:
    fn = fname.lower()
    return any(fn.endswith(ext) for ext in ARCHIVE_EXT)

def pick_mesh_file(root: Path) -> Optional[Path]:
    """
    在 root（及子目录）里按优先级挑一个可渲染网格文件。
    """
    candidates: List[Path] = []
    for ext in MESH_EXT_PRIORITY:
        matches = list(root.rglob(f"*{ext}"))
        if matches:
            # 优先挑路径短/深度浅的
            matches.sort(key=lambda p: (len(p.parts), len(str(p))))
            return matches[0]
    return None

def has_existing_model(shape_dir: Path) -> bool:
    raw_dir = shape_dir / "raw"
    if not raw_dir.is_dir():
        return False
    # 查找常见 3D 文件
    for ext in ["*.glb", "*.fbx", "*.obj"]:
        if list(raw_dir.glob(ext)):
            return True
    return False

# ---------- 下载器：严格失败 + 采集响应信息 ----------
import os, subprocess, tempfile, shutil, sys
from pathlib import Path

def download_with_meta(url: str, dst: Path, overwrite: bool=False) -> dict:
    """
    严格下载，返回 {ok, http_code, url_effective, bytes, err}.
    使用 curl 时加 -f（HTTP错误不落盘），失败情况下不会留下空文件。
    """
    import os, subprocess, tempfile, shutil
    dst = Path(dst)
    if dst.exists() and not overwrite:
        return {"ok": True, "http_code": 200, "url_effective": url, "bytes": dst.stat().st_size, "err": None}

    curl = shutil.which("curl")
    if curl:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            head = tf.name
        try:
            cmd = [curl, "-f", "-L", "-sS", "-D", head, "-o", str(dst), url]  # 注意 -f
            cp = subprocess.run(cmd, capture_output=True, text=True)
            if cp.returncode != 0:
                http_code = None
                try:
                    htxt = Path(head).read_text(errors="ignore")
                    for line in htxt.splitlines():
                        if line.startswith("HTTP/"):
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[1].isdigit():
                                http_code = int(parts[1]); break
                except Exception:
                    pass
                if dst.exists() and dst.stat().st_size == 0:
                    try: dst.unlink()
                    except Exception: pass
                return {"ok": False, "http_code": http_code, "url_effective": url, "bytes": 0,
                        "err": (cp.stderr or cp.stdout or "").strip() or "curl failed"}
            
            http_code = None
            try:
                htxt = Path(head).read_text(errors="ignore")
                for line in reversed(htxt.splitlines()):
                    if line.startswith("HTTP/"):
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            http_code = int(parts[1]); break
            except Exception:
                pass
            return {"ok": True, "http_code": http_code or 200, "url_effective": url,
                    "bytes": dst.stat().st_size, "err": None}
        finally:
            try: os.remove(head)
            except Exception: pass

    # 退化：urllib
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "wb") as f: f.write(data)
            return {"ok": True, "http_code": getattr(resp, "status", None) or 200,
                    "url_effective": resp.geturl(), "bytes": len(data), "err": None}
    except Exception as e:
        return {"ok": False, "http_code": None, "url_effective": url, "bytes": 0, "err": str(e)}


# --- 兼容：保留原 safe_download 接口（返回 Path 或 None），不再返回 dict ---
def safe_download(url: str, dst: Path, overwrite: bool=False):
    """
    兼容旧用法。写文件到 dst。成功时返回 dst（或保持与旧版一致：返回 None），失败抛异常。
    """
    meta = download_with_meta(url, dst, overwrite=overwrite)
    if not meta["ok"]:
        raise RuntimeError(f"download failed: code={meta.get('http_code')} url={meta.get('url_effective')} err={meta.get('err')}")
    return dst  # 如果你旧版是返回 None，这里也可以改成 return None

# ---------- 校验器：返回 (ok, reason, detail) ----------
def _check_download_reason_inline(path: Path, expect_ext: str|None=None) -> tuple[bool, str, str]:
    """
    根据文件内容判断是否为“有效图片”，并给出原因码：
    ok, html_error, lfs_pointer, empty_file, too_small, bad_magic_png/jpg/.../unknown
    detail 为简短补充信息。
    """
    try:
        b = path.read_bytes()
    except Exception as e:
        return False, "empty_file", f"read_error:{e}"

    if not b:
        return False, "empty_file", "size=0"

    head = b[:512].lower()
    # HTML/文本错误页
    html_markers = (b"<html", b"<!doctype", b"not found", b"forbidden", b"access denied", b"sign in", b"github", b"error")
    if any(m in head for m in html_markers):
        return False, "html_error", "looks_like_html_or_error_page"

    # LFS 指针文件
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False, "lfs_pointer", "git_lfs_pointer_file"

    # 过小（一般不是正常图片）
    if len(b) < 64:
        return False, "too_small", f"size={len(b)}"

    # 魔数
    sig = b[:16]
    def starts(*prefixes): return any(sig.startswith(p) for p in prefixes)

    # PNG
    if starts(b"\x89PNG\r\n\x1a\n"):
        return True, "ok", "png"
    # JPEG
    if sig[:2] == b"\xFF\xD8":
        return True, "ok", "jpg"
    # DDS
    if starts(b"DDS "):
        return True, "ok", "dds"
    # KTX(1/2)
    if starts(b"\xABKTX 11\xBB\r\n\x1A\n") or starts(b"\xABKTX 22\xBB\r\n\x1A\n"):
        return True, "ok", "ktx"
    # EXR
    if starts(b"\x76\x2F\x31\x01"):
        return True, "ok", "exr"
    # TGA/BMP（宽松判断）
    if sig[:2] in (b"\x00\x00", b"BM"):
        # 不能 100% 保证，但到这里一般可放行
        return True, "ok", "tga_or_bmp_or_unknown_but_binary"

    # 期望后缀但魔数不对
    if expect_ext:
        return False, f"bad_magic_{expect_ext.lower().strip('.')}", "signature_mismatch"

    return False, "bad_magic_unknown", "signature_mismatch"


GOOD_MAGIC = {
    "gz": b"\x1f\x8b",
    "zip": b"PK\x03\x04",
    "glb": b"glTF",   # b"glTF"
    # 其它格式：OBJ/PLY/STL 多为 ASCII，无固定魔数，靠后续解压/解析再查
}
BAD_MAGIC_PREFIX = (b"GIF8", b"<!DOCT", b"<html", b"<?xml")  # GIF/HTML/XML错误页

def head_bytes(path, n=8):
    with open(path, "rb") as f:
        return f.read(n)

def looks_like_asset(path: Path) -> bool:
    hb = head_bytes(path, 8)
    if hb.startswith(BAD_MAGIC_PREFIX):
        return False
    # 若是压缩包或 glb，有明确魔数
    name = path.name.lower()
    if name.endswith((".tar.gz", ".tgz")):   # tar.gz 里面是 tar，无固定魔数；但.gz 有 1f 8b
        return hb.startswith(GOOD_MAGIC["gz"])
    if name.endswith(".zip"):
        return hb.startswith(GOOD_MAGIC["zip"])
    if name.endswith(".glb"):
        return hb.startswith(GOOD_MAGIC["glb"])
    # 其它（.obj/.ply/.stl/.fbx/.blend）不靠魔数，放行，后续再用“能否解析/是否找到网格文件”判断
    return True

def fix_github_blob(url: str) -> str:
    # import re
    # m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    # if not m:
    #     return url
    # u, r, sha, p = m.groups()
    # from urllib.parse import quote
    # return f"https://raw.githubusercontent.com/{u}/{r}/{sha}/" + "/".join(quote(x) for x in p.split("/"))
    import re
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if not m:
        return url
    u, r, sha, p = m.groups()
    from urllib.parse import quote, unquote
    rel = "/".join(quote(seg, safe="") for seg in unquote(p).split("/") if seg)
    return f"https://raw.githubusercontent.com/{u}/{r}/{sha}/{rel}"
# === Inline helpers for OBJ parent-dir selective fetch + precise fill ===
# 可调参数：
IMG_EXT_INLINE = (".png",".jpg",".jpeg",".tga",".tif",".tiff",".exr",".bmp",".dds",".ktx",".ktx2",".webp")
# 同名必抓扩展（含 .png.meta/.jpg.meta 等）：
SAME_NAME_EXTRA_EXTS_INLINE = (".mtl",".json",".meta", ".png", ".jpg", ".urdf", ".ply") + tuple(ext + ".meta" for ext in IMG_EXT_INLINE)
# 常见材质子目录（只递归一层）：
SUBDIR_CANDIDATES_INLINE = ("Textures","textures","Materials","materials","images","Images","maps","Maps")
# 少量图片“一把梭”的阈值：
IMG_BULK_THRESHOLD_INLINE = 10

# def _base_url_inline(url: str) -> str:
#     sp = urlsplit(url)
#     path = sp.path
#     path = (path.rsplit("/", 1)[0] + "/") if "/" in path else "/"
#     return urlunsplit((sp.scheme, sp.netloc, path, "", ""))
# 1）统一一段工具
from urllib.parse import quote, unquote
def _encode_repo_path(p: str) -> str:
    return "/".join(quote(unquote(seg), safe="") for seg in (p or "").split("/") if seg)

def _join_url_dir_inline(base: str, rel: str) -> str:
    from urllib.parse import urljoin, quote, unquote
    rel_decoded = unquote(rel)                  # 先把 %20 还原为空格
    parts = [quote(p, safe="") for p in rel_decoded.split("/") if p]
    return urljoin(base, "/".join(parts))       # 再编码一次，避免二次编码

def _http_fetch_text_inline(url: str, timeout: float = 15.0) -> str | None:
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return None

def _github_parse_url_inline(url: str):
    """Return (user, repo, sha, dir_path, filename) or None; supports github.com/blob and raw.githubusercontent.com."""
    import re
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
    if m:
        u, r, sha, p = m.groups(); parts = p.split("/")
        return u, r, sha, "/".join(parts[:-1]), parts[-1]
    m = re.match(r"https?://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.*)", url)
    if m:
        u, r, sha, p = m.groups(); parts = p.split("/")
        return u, r, sha, "/".join(parts[:-1]), parts[-1]
    return None

def _github_list_files_in_dir_inline(user: str, repo: str, sha: str, dir_path: str) -> list[str]:
    """List direct children files (no recursion) under a GitHub tree dir, return repo-root relative paths."""
    import re
    # page_url = f"https://github.com/{user}/{repo}/tree/{sha}/" + (dir_path + "/" if dir_path else "")
    page_url = f"https://github.com/{user}/{repo}/tree/{sha}/" + ((_encode_repo_path(dir_path) + "/") if dir_path else "")
    html = _http_fetch_text_inline(page_url) or ""
    files = []
    for href in re.findall(r'href="(/[^"]+/blob/[^"]+)"', html):
        href = href.split('?', 1)[0]  # strip ?plain=1 等
        m = re.match(rf'^/{re.escape(user)}/{re.escape(repo)}/blob/{re.escape(sha)}/(.+)$', href)
        if m: files.append(m.group(1))
    out, seen = [], set()
    for f in files:
        if f not in seen: out.append(f); seen.add(f)
    return out

import collections, re
from urllib.parse import quote, unquote

# 放在你现有的 GitHub 辅助函数附近（不会影响原函数）
import collections, re
GITHUB_ENUM_MAX_DEPTH = 3 if not FULL_REPO_SEARCH else 64     # 默认2层；全仓库时放大
GITHUB_ENUM_MAX_VISITS = 500 if not FULL_REPO_SEARCH else 5000  # 防爆，仍设上限

def _github__fetch_tree_html_inline(user: str, repo: str, sha: str, dir_path: str) -> str:
    # page_url = f"https://github.com/{user}/{repo}/tree/{sha}/" + (dir_path + "/" if dir_path else "")
    page_url = f"https://github.com/{user}/{repo}/tree/{sha}/" + ((_encode_repo_path(dir_path) + "/") if dir_path else "")
    return _http_fetch_text_inline(page_url) or ""

def _github_tree_api_list_all_files_inline(user: str, repo: str, sha: str) -> list[str]:
    """用 GitHub Tree API 列出指定 sha 的全部文件路径（可能被截断，但通常比 HTML 更全）"""
    key = (user, repo, sha)
    if key in _GITHUB_TREE_CACHE:
        return _GITHUB_TREE_CACHE[key]
    import json, urllib.request
    api_url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{sha}?recursive=1"
    req = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "fbx-texture-fetcher/1.0",
            **({"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}),
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        tree = data.get("tree", []) or []
        files = [e.get("path") for e in tree if isinstance(e, dict) and e.get("type") == "blob" and e.get("path")]
        _GITHUB_TREE_CACHE[key] = files
        if _GITHUB_TREE_CACHE_FILE:
            _git_tree_cache_save(_GITHUB_TREE_CACHE_FILE)

        return files
    except Exception:
        return []

def _github_api_find_raw_urls_by_filename_inline(user: str, repo: str, sha: str, filename: str, limit: int = 8) -> list[str]:
    """在该 sha 的文件树里按 basename 匹配，返回 raw 链接"""
    from pathlib import Path as _Path
    files = _github_tree_api_list_all_files_inline(user, repo, sha)
    if not files:
        return []
    low = filename.lower()
    hits = [p for p in files if _Path(p).name.lower() == low][:max(1, limit)]
    urls = []
    for rp in hits:
        urls.append(_github_raw_url_inline(user, repo, sha, rp))
    return urls

def _github_list_files_recursive_inline(user: str, repo: str, sha: str, start_dir: str, max_depth:int=GITHUB_ENUM_MAX_DEPTH) -> list[str]:
    """从 start_dir 起 BFS 递归 enumer，返回仓库根相对路径（文件）。"""
    out, seen_files = [], set()
    q = collections.deque([(start_dir or "", 0)])
    visits = 0
    while q:
        d, depth = q.popleft()
        if visits >= GITHUB_ENUM_MAX_VISITS:
            break
        html = _github__fetch_tree_html_inline(user, repo, sha, d); visits += 1

        # 文件（/blob/）
        for href in re.findall(r'href="(/[^"]+/blob/[^"]+)"', html):
            href = href.split("?", 1)[0]
            m = re.match(rf"^/{re.escape(user)}/{re.escape(repo)}/blob/{re.escape(sha)}/(.+)$", href)
            if m:
                rp = m.group(1)
                if rp not in seen_files:
                    out.append(rp); seen_files.add(rp)

        # 目录（/tree/）
        if depth < max_depth:
            for dh in re.findall(r'href="(/[^"]+/tree/[^"]+)"', html):
                dh = dh.split("?", 1)[0]
                m = re.match(rf"^/{re.escape(user)}/{re.escape(repo)}/tree/{re.escape(sha)}/(.+)$", dh)
                if m:
                    child = m.group(1)
                    if child != d:
                        q.append((child, depth+1))
    return out

def _github_find_search_inline(user, repo, sha, filename):
    url = f"https://github.com/{user}/{repo}/search?q={quote('filename:' + filename)}&type=code"
    html = _http_fetch_text_inline(url) or ""
    hits = []

    # Phase-1: 只接受与 sha 完全一致的结果（与你原本相同行为）
    for href in re.findall(
        rf'href="/{re.escape(user)}/{re.escape(repo)}/blob/{re.escape(sha)}/([^"#]+)"',
        html
    ):
        if Path(href).name.lower() == filename.lower():
            hits.append(href)

    if hits:
        # 去重并保持顺序
        return list(dict.fromkeys(hits))

    # Phase-2: 放宽到任何 ref（main/tag/其他 sha），仅用于提示“仓库别的版本存在”
    for ref, path in re.findall(
        rf'href="/{re.escape(user)}/{re.escape(repo)}/blob/([^/]+)/([^"#]+)"',
        html
    ):
        if Path(path).name.lower() == filename.lower():
            hits.append(path)

    return list(dict.fromkeys(hits))  # 仍返回“相对路径列表”


# —— 有效性校验（防止 HTML/404/占位符）——
def _looks_like_html_err(head: bytes, text: str) -> bool:
    if head.startswith(b"<!DOCTYPE") or head.lstrip().startswith(b"<html"): 
        return True
    kw = ("<html", "not found", "404", "error", "forbidden", "login", "sign in", "github")
    t = (text or "").lower()
    return any(k in t for k in kw)

def _valid_image_magic(p: Path, ext: str, head: bytes) -> bool:
    if ext==".png":   return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in (".jpg",".jpeg"): return head.startswith(b"\xff\xd8")
    if ext in (".tif",".tiff"): return head[:4] in (b"II*\x00", b"MM\x00*")
    if ext==".dds":   return head.startswith(b"DDS ")
    if ext in (".ktx",".ktx2"): return head.startswith(b"\xabKTX 1") or head.startswith(b"\xabKTX 2")
    if ext==".exr":   return head[:4] == b"\x76\x2f\x31\x01"
    # 其它如 .tga 等魔数不稳定，放行
    return True

def _is_valid_downloaded_inline(p: Path) -> bool:
    try:
        if not p.exists() or p.stat().st_size == 0: 
            return False
        ext = p.suffix.lower()
        head = p.read_bytes()[:64]
        try:
            txt_head = p.read_text(errors="ignore")[:256]
        except Exception:
            txt_head = ""
        if _looks_like_html_err(head, txt_head): 
            return False
        if ext in IMG_EXT_INLINE:
            return _valid_image_magic(p, ext, head)
        if ext == ".json":
            import json; json.loads(p.read_text(errors="ignore")); return True
        if ext == ".mtl":
            t = p.read_text(errors="ignore")[:4096].lower()
            if "<html" in t: return False
            return ("newmtl" in t) or ("map_" in t)
        if ext in (".meta", ".import"):
            return "<html" not in txt_head.lower()
        return True
    except Exception:
        return False

def _check_download_reason_inline(path: Path, expect_ext: str|None=None) -> tuple[bool, str]:
    """
    返回 (ok, reason)。reason 可能是：
    ok / html_error / lfs_pointer / empty_file / too_small / bad_magic_png/jpg/.../unknown
    """
    try:
        b = path.read_bytes()
    except Exception:
        return False, "empty_file"

    if not b:
        return False, "empty_file"
    head = b[:512].lower()

    if any(m in head for m in (b"<html", b"<!doctype", b"not found", b"forbidden", b"access denied", b"sign in", b"github", b"error")):
        return False, "html_error"
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False, "lfs_pointer"
    if len(b) < 64:
        return False, "too_small"

    sig = b[:16]
    if sig.startswith(b"\x89PNG\r\n\x1a\n"): return True, "ok"
    if sig[:2] == b"\xFF\xD8": return True, "ok"
    if sig.startswith(b"DDS "): return True, "ok"
    if sig.startswith(b"\xABKTX 11\xBB\r\n\x1A\n") or sig.startswith(b"\xABKTX 22\xBB\r\n\x1A\n"): return True, "ok"
    if sig.startswith(b"\x76\x2F\x31\x01"): return True, "ok"
    if sig[:2] in (b"\x00\x00", b"BM"): return True, "ok"

    if expect_ext:
        return False, f"bad_magic_{expect_ext.lower().strip('.')}"
    return False, "bad_magic_unknown"

# def _parse_mtl_for_textures_inline(mtl_path: Path) -> list[str]:
#     import re
#     tex, keys = [], ("map_Kd","map_Ks","map_Ns","map_Bump","bump","disp","refl","norm",
#                      "map_Pr","map_Pm","map_Ps","map_AO","map_d","map_Ke")
#     try:
#         for line in mtl_path.read_text(errors="ignore").splitlines():
#             s = line.strip()
#             if not s or s.startswith("#"): continue
#             if any(s.lower().startswith(k.lower()) for k in keys):
#                 parts = line.split()
#                 for i in range(len(parts)-1, -1, -1):
#                     if not parts[i].startswith("-"):
#                         candidate = parts[i]; break
#                 else:
#                     continue
#                 lc = candidate.lower()
#                 if any(lc.endswith(ext) for ext in IMG_EXT_INLINE):
#                     tex.append(candidate)
#     except Exception:
#         pass
#     out, seen = [], set()
#     for n in tex:
#         if n not in seen: out.append(n); seen.add(n)
#     return out

################ obj/mtl 解析相关 ################
import shlex, re
from pathlib import Path

def _obj_find_mtllibs_inline(obj_text: str) -> list[str]:
    libs = []
    for m in re.finditer(r'(?im)^\s*mtllib\s+(.+?)\s*$', obj_text):
        # 允许一行多个 mtllib，支持引号/空格
        parts = shlex.split(m.group(1))
        for p in parts:
            p = p.strip()
            if not p:
                continue
            libs.append(p.replace("\\", "/"))
    # 去重保序
    return list(dict.fromkeys(libs))

import re
FBX_TEX_EXTS = (".png",".jpg",".jpeg",".tga",".tiff",".bmp",".exr",".dds")

import shlex
from urllib.parse import urljoin

# 认为是贴图的扩展名
OBJ_TEX_EXTS = (".png", ".jpg", ".jpeg", ".tga", ".tiff", ".bmp", ".exr", ".dds")

# MTL 里常见的贴图指令（按 glTF/PBR 兼容尽量多列）
MTL_MAP_KEYS = {
    "map_Kd", "map_Ks", "map_Ka", "map_d", "map_bump", "bump", "disp", "norm",
    "map_Ke",  # emissive
    "map_Pr",  # roughness (some exporters)
    "map_Pm",  # metallic
    "map_Ps",  # sheen
}

def _normalize_relpath(p: str) -> str:
    p = str(p).strip().strip('"').strip("'").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    # ★ 新增：Windows 盘符/UNC 直接扁平化为 basename
    import re
    if re.match(r"^[A-Za-z]:/", p) or p.startswith("//") or p.startswith("\\\\"):
        return Path(p).name
    # ★ 新增：越级一律拒绝，留 basename
    if "/../" in f"/{p}":
        return Path(p).name
    return p

def parse_obj_for_mtllibs(obj_path: Path) -> list[Path]:
    mtls: list[Path] = []
    try:
        with open(obj_path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("mtllib"):
                    rest = raw.split(None, 1)[1] if " " in raw else ""
                    parts = shlex.split(rest, posix=True)
                    buf = []
                    for p in parts:
                        buf.append(p)
                        joined = " ".join(buf)
                        if joined.lower().endswith(".mtl"):
                            nm = _normalize_relpath(joined)
                            if nm:
                                mtls.append(Path(nm))
                            buf = []
                    # 兜底：少数奇葩行最后再收一次
                    if buf:
                        joined = " ".join(buf)
                        if joined.lower().endswith(".mtl"):
                            nm = _normalize_relpath(joined)
                            if nm:
                                mtls.append(Path(nm))
    except Exception as e:
        print(f"    ⚠ parse_obj_for_mtllibs error: {e}")
    # 去重保序
    return list(dict.fromkeys(mtls))

import difflib, re

def _norm_base(s: str) -> str:
    b = Path(s).stem.lower()
    return re.sub(r"[\s_\-\.]+", "", b)

def autofix_obj_mtllib(obj_path: Path) -> Path | None:
    """若 mtllib 指向的 .mtl 在本地缺失，则在同目录猜测最佳 .mtl 并重写 _fixed.obj。返回新的 obj 路径或 None（无改动）。"""
    mtls_decl = [m.name for m in parse_obj_for_mtllibs(obj_path)]
    if not mtls_decl:
        # 没有 mtllib，不处理
        return None

    folder = obj_path.parent
    local_mtls = [p.name for p in folder.glob("*.mtl")]
    if not local_mtls:
        return None

    obj_base = obj_path.stem
    obj_norm = _norm_base(obj_base)
    local_norm_map = {n: _norm_base(n) for n in local_mtls}

    # 仅修“缺失”的声明项
    missing = [m for m in mtls_decl if not (folder / m).exists()]
    if not missing:
        return None

    # 构造替换表：缺失声明 => 选中的本地 .mtl
    replace_map: dict[str,str] = {}
    for miss in missing:
        miss_norm = _norm_base(miss)
        # 1) 优先 <obj名>.mtl
        preferred = f"{obj_base}.mtl"
        if preferred in local_mtls:
            replace_map[miss] = preferred
            continue
        # 2) 归一化相等
        eq = next((n for n, nb in local_norm_map.items() if nb == miss_norm or nb == obj_norm), None)
        if eq:
            replace_map[miss] = eq
            continue
        # 3) 相似度最高者（>=0.8）
        cand = max(local_mtls, key=lambda n: difflib.SequenceMatcher(None, miss_norm, local_norm_map[n]).ratio())
        if difflib.SequenceMatcher(None, miss_norm, _norm_base(cand)).ratio() >= 0.8:
            replace_map[miss] = cand
        # 若仍没有合适项，就放弃该 miss（保留原样）

    if not replace_map:
        return None

    # —— 生成 _fixed.obj：把 mtllib 行重写
    text = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    for line in text:
        if line.lower().startswith("mtllib"):
            # 按“累积到 .mtl”方式逐个替换（与 parse 保持一致）
            rest = line.split(None, 1)[1] if " " in line else ""
            parts = shlex.split(rest, posix=True)
            buf, rebuilt = [], []
            for p in parts:
                buf.append(p)
                joined = " ".join(buf)
                if joined.lower().endswith(".mtl"):
                    old = joined
                    new = replace_map.get(old, old)
                    rebuilt.append(new)
                    buf = []
            if buf:  # 兜底
                joined = " ".join(buf)
                if joined.lower().endswith(".mtl"):
                    rebuilt.append(replace_map.get(joined, joined))
            if rebuilt:
                line = "mtllib " + " ".join(rebuilt)
        out.append(line.replace("\\", "/"))  # 顺便统一斜杠
    fixed = obj_path.with_name(obj_path.stem + "_fixed.obj")
    fixed.write_text("\n".join(out), encoding="utf-8")
    print(f"    · autofixed mtllib → {fixed.name}: {replace_map}")
    return fixed

def parse_mtl_for_textures(mtl_path: Path) -> set[Path]:
    """解析一个 MTL，收集所有 map_* 引用（仅取最后一个非选项 token 作为贴图路径）"""
    refs: set[Path] = set()
    try:
        with open(mtl_path, "r", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                key = line.split(maxsplit=1)[0]
                if key not in MTL_MAP_KEYS:
                    continue
                # 用 shlex 解析整行，处理引号/空格/转义
                toks = shlex.split(line, posix=True)
                # 丢掉 key 本身
                toks = toks[1:]
                if not toks:
                    continue
                # 跳过以 '-' 开头的选项（如 -bm 0.3 -clamp on ...），取最后一个非选项 token
                path_tok = None
                for t in reversed(toks):
                    if t.startswith("-"):
                        continue
                    path_tok = t
                    break
                if not path_tok:
                    continue
                rel = _normalize_relpath(path_tok)
                if not rel:
                    continue
                # NEW: Windows 绝对/UNC 直接取 basename
                if re.match(r"^[a-zA-Z]:/", rel) or rel.startswith("//") or rel.startswith("\\\\"):
                    rel = Path(rel).name
                # 只接受常见贴图扩展（有些导出器会把 .exr/.dds 放进来）
                if not any(rel.lower().endswith(ext) for ext in OBJ_TEX_EXTS):
                    # 仍然加入，后面下载失败就记失败，不强行过滤
                    pass
                refs.add(Path(rel))
    except Exception as e:
        print(f"    ⚠ parse_mtl_for_textures error ({mtl_path}): {e}")
    return refs

_MTLMAP_KEYS = {
    "map_ka","map_kd","map_ks","map_ke","map_ns","map_d",
    "bump","map_bump","map_bump","norm","disp","decal","refl"
}

def _mtl_find_texture_paths_inline(mtl_text: str) -> list[str]:
    tex = []
    for raw in mtl_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split(None, 1)[0].lower()
        if key not in _MTLMAP_KEYS:
            continue
        rest = line[len(key):].strip()
        if not rest:
            continue
        # 用 shlex 解析带引号的路径；文件名通常是“最后一个非选项 token”
        tokens = shlex.split(rest)
        if not tokens:
            continue
        # 过滤掉选项参数，保留最后一个非选项串作为文件名/URL
        # 大多数实现中文件名最后；为简洁起见，取最后一个 token
        cand = tokens[-1].replace("\\", "/")
        # 行内注释清理（若作者把 # 写在路径后面）
        cand = cand.split("#", 1)[0].strip()
        if cand:
            tex.append(cand)
    return list(dict.fromkeys(tex))

def _expand_udim_candidates_inline(name: str) -> list[str]:
    out = [name]
    low = name.lower()

    # 1) <UDIM> pattern
    if "<udim>" in low:
        out.append(name)  # 保留原始（以防仓库直接存成<UDIM>）
        # 真正展开在“仓库匹配”环节做：将 <UDIM> → r'(\d{4})' 的 regex 去 Tree API 匹配

    # 2) Mari u1_v1
    m = re.search(r'[._-]u(\d+)_v(\d+)', name, flags=re.I)
    if m:
        u, v = int(m.group(1)), int(m.group(2))
        udim = 1001 + (u - 1) + 10 * (v - 1)
        out.append(name.replace(m.group(0), f".{udim}"))

    return list(dict.fromkeys(out))

def _github_api_find_raw_urls_by_regex_inline(user: str, repo: str, sha: str, filename_regex: str, limit: int = 64) -> list[str]:
    files = _github_tree_api_list_all_files_inline(user, repo, sha) or []
    r = re.compile(filename_regex, flags=re.I)
    hits = []
    for p in files:
        basename = Path(p).name
        if r.fullmatch(basename):
            hits.append(_github_raw_url_inline(user, repo, sha, p))
            if len(hits) >= limit:
                break
    return hits
import re
from pathlib import Path

COMMON_TEX_EXTS = [".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tiff", ".webp", ".dds"]

def filename_variants_exact_first(filename: str) -> list[str]:
    """
    生成用于仓库搜索的候选文件名列表：
    - 第1个永远是“原名，含扩展名”（精确匹配）
    - 然后才是大小写与下划线/连字符/空格的互换变体（仍保留扩展名）
    """
    base = Path(filename).name
    stem = Path(base).stem
    ext  = Path(base).suffix

    out = []
    seen = set()

    def push(x: str):
        if x not in seen:
            out.append(x); seen.add(x)

    # 1) 原名优先（关键！）
    push(base)

    # 2) 大小写变体
    for v in {base.lower(), base.upper(), base.title()}:
        push(v)

    # 3) 仅对 stem 做 _ / - / 空格 的互换；扩展名保留
    combos = {
        stem.replace("_", "-"),
        stem.replace("-", "_"),
        stem.replace("_", " "),
        stem.replace(" ", "_"),
        stem.replace("-", " "),
        stem.replace(" ", "-"),
    }
    for s in combos:
        for v in {s, s.lower(), s.upper(), s.title()}:
            push(v + ext)

    return out

# def ensure_complete_obj_asset_strict(obj_file: Path, src_url: str, raw_dir: Path, overwrite=False) -> tuple[int,int,int,list[str]]:
#     """
#     更稳的 OBJ 补齐：
#     - 找 mtllib（若没有，尝试就地扫描 *.mtl）
#     - 逐个 MTL 解析 map_* 指令
#     - 构造上游 url 前缀，尝试下载相对路径指向的贴图到 raw/ 下相对目录
#     返回: (下载成功数, 失败数, 已存在跳过数, 成功文件名列表)
#     """
#     # 在 ensure_complete_obj_asset_strict(...) / ensure_complete_fbx_asset(...) 内部
#     def _try_fetch_one(url_dst):
#         url, dst = url_dst
#         try:
#             safe_download(url, dst, overwrite=overwrite)
#             return (dst.name, True, url, None)
#         except Exception as e:
#             return (dst.name, False, url, str(e))
        
#     ok = fail = skipped = 0
#     names: list[str] = []

#     # 1) 找到所有 MTL
#     mtls = parse_obj_for_mtllibs(obj_file)
#     if not mtls:
#         # 没有显式 mtllib，就尝试旁边有没有 *.mtl
#         mtls = [p.name for p in obj_file.parent.glob("*.mtl")]

#     # 2) 汇总所有 MTL 内的纹理相对路径
#     tex_refs: set[Path] = set()
#     for m in mtls:
#         mtl_local = (obj_file.parent / m).resolve()
#         if not mtl_local.exists():
#             # 若本地没有 mtl，先尝试把它下载下来（和 OBJ 同上游目录）
#             try:
#                 base_url = re.sub(r"[^/]+$", "", src_url)
#                 # mtl_url = urljoin(base_url, _normalize_relpath(str(m)))
#                 mtl_url = _join_url_dir_inline(base_url, _normalize_relpath(str(m)))
#                 dst_mtl = raw_dir / m
#                 dst_mtl.parent.mkdir(parents=True, exist_ok=True)
#                 safe_download(mtl_url, dst_mtl, overwrite=overwrite)
#                 mtl_local = dst_mtl
#                 print(f"    · fetched missing MTL: {m}")
#             except Exception as e:
#                 print(f"    ⚠ missing MTL, and fetch failed: {m} ({e})")
#                 continue
#         tex_refs |= parse_mtl_for_textures(mtl_local)

#     if not tex_refs:
#         print("    · no texture refs in MTL (or MTL missing)")
#         return (0, 0, 0, [])

#     # 3) 按相对路径补齐下载
#     base_url = re.sub(r"[^/]+$", "", src_url)  # OBJ 上游同目录
#     base_url = fix_github_blob(base_url)       # 你的工具：blob→raw
#     for rel in sorted(tex_refs, key=lambda p: p.as_posix().lower()):
#         rel_norm = Path(_normalize_relpath(rel.as_posix()))
#         dst = raw_dir / rel_norm
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         if dst.exists() and not overwrite:
#             skipped += 1
#             continue
#         try:
#             # url = urljoin(base_url, rel_norm.as_posix())
#             url = _join_url_dir_inline(base_url, rel_norm.as_posix())
#             safe_download(url, dst, overwrite=overwrite)
#             ok += 1; names.append(rel_norm.as_posix())
#         except Exception as e:
#             print(f"    ⚠ OBJ texture fetch fail: {rel_norm} ← {url} ({e})")
#             fail += 1

#         gh = _github_parse_url_inline(base_url)  # 若来源是 GitHub 则返回 (user, repo, sha, dir, filename)
#         if gh:
#             user, repo, sha, dir_path, _ = gh
#             fname = Path(rel_norm.name).name
#             # 1) 就近目录 & 常见子目录里按文件名寻找
#             raw_urls = _github_search_texture_in_repo_inline(user, repo, sha, dir_path, fname,
#                                                             subdirs=SUBDIR_CANDIDATES_INLINE)
#             # 2) 若仍未命中，递归 2~3 层兜底（覆盖 “Materials and Textures” 这类非常规目录）
#             if not raw_urls:
#                 for rp in _github_list_files_recursive_inline(user, repo, sha, dir_path,
#                                                             max_depth=GITHUB_ENUM_MAX_DEPTH):
#                     if Path(rp).name.lower() == fname.lower():
#                         raw_urls.append(_github_raw_url_inline(user, repo, sha, rp))
#                         break
#             # 3) 逐个候选再试下载
#             for alt in raw_urls:
#                 try:
#                     safe_download(alt, dst, overwrite=overwrite)
#                     ok += 1; names.append(rel_norm.as_posix())
#                     print(f"    · OBJ texture rescued via repo search: {fname} ← {alt}")
#                     break
#                 except Exception:
#                     continue
#             else:
#                 print(f"    ⚠ OBJ texture still missing after repo search: {fname}")

#     return (ok, fail, skipped, names)
def ensure_complete_obj_asset_strict(obj_file: Path, src_url: str, raw_dir: Path, overwrite=False,
                                     local_only: bool=False, local_roots: list[Path] | None=None) -> tuple[int,int,int,list[str]]:
    """
    更稳/更快的 OBJ 补齐（并发 + 终态日志）：
    - 找 mtllib（若没有，尝试同目录 *.mtl）
    - 解析所有 MTL 的 map_* 引用
    - 先用“同目录直链”尝试下载；失败再用 GitHub 仓库就近/递归搜索兜底
    - 并发执行每张贴图的拉取；最终只打印一次“成功/失败/跳过”的结果
    返回: (下载成功数, 失败数, 已存在跳过数, 成功文件名列表相对路径)
    """
    fixed = autofix_obj_mtllib(obj_file)
    if fixed and fixed.exists():
        obj_file = fixed  # 改为使用 _fixed.obj

    local_roots = [p for p in (local_roots or []) if isinstance(p, Path) and p.exists()]
    ok = fail = skipped = 0
    names: list[str] = []

    # 1) 找到所有 MTL
    mtls = parse_obj_for_mtllibs(obj_file) or [p.name for p in obj_file.parent.glob("*.mtl")]
    if not mtls:
        mtls = [p.name for p in obj_file.parent.glob("*.mtl")]
    # ---- NEW: 本地优先 ----
    # 1) 拷贝缺失的 .mtl 到 raw/
    for m in mtls:
        dst_mtl = raw_dir / Path(m).name
        if not dst_mtl.exists() or overwrite:
            src_mtl = (obj_file.parent / m)
            if not src_mtl.exists():
                # 在 local_roots 查找
                src_mtl = _local_find_one(Path(m), local_roots) or _local_find_one(Path(Path(m).name), local_roots)
            if src_mtl and src_mtl.exists():
                try:
                    shutil.copy2(src_mtl, dst_mtl)
                    ok += 1; names.append(dst_mtl.name)
                except Exception:
                    fail += 1
            else:
                # 没有也无所谓，后面贴图可能仍然可补
                pass

    # 2) 解析 MTL 里的贴图引用并在本地 roots 查找 → 复制到 raw/
    # tex_refs: set[Path] = set()
    # for m in mtls:
    #     loc = (raw_dir / Path(m).name)
    #     if not loc.exists():
    #         loc = (obj_file.parent / m)
    #     if loc.exists():
    #         tex_refs |= parse_mtl_for_textures(loc)

    # for rel in sorted(tex_refs, key=lambda p: p.as_posix().lower()):
    #     # rel_norm = Path(_normalize_relpath(rel))
    #     # dst = raw_dir / rel_norm # save with relative path
    #     # dst = raw_dir / Path(rel).name
    #     rel_norm = Path(_normalize_relpath(rel.as_posix()))  # <== 关键：Path→str→normalize
    #     dst = (raw_dir / rel_norm).resolve()
    #     dst.parent.mkdir(parents=True, exist_ok=True)        # <== 关键：先建目录
    #     if dst.exists() and not overwrite:
    #         skipped += 1
    #         continue
    #     src = _local_find_one(rel, local_roots) or _local_find_one(Path(rel.name), local_roots)
    #     if src:
    #         try:
    #             shutil.copy2(src, dst)
    #             ok += 1; names.append(dst.name)
    #         except Exception:
    #             fail += 1

    # if local_only:
    #     # === 本地检索统计与打印 ===
    #     # 1) 统计已放置（raw_dir 下已存在）的文件
    #     placed = []
    #     for m in mtls:
    #         mtl_dst = raw_dir / Path(m).name
    #         if mtl_dst.exists():
    #             placed.append(mtl_dst.name)

    #     placed_tex = []
    #     missing_tex = []
    #     for rel in sorted(tex_refs, key=lambda p: p.as_posix().lower()):
    #         bn = Path(rel).name
    #         if (raw_dir / bn).exists():
    #             placed_tex.append(bn)
    #         else:
    #             missing_tex.append(rel.as_posix())
    # 2) 解析 MTL 里的贴图引用并在本地 roots 查找 → 复制到 raw/
    tex_refs: set[Path] = set()
    for m in mtls:
        loc = (raw_dir / Path(m).name)
        if not loc.exists():
            loc = (obj_file.parent / m)
        if loc.exists():
            tex_refs |= parse_mtl_for_textures(loc)

    placement_map: dict[str, str] = {}  # basename -> 相对路径（用于后续改写 mtl）

    for rel in sorted(tex_refs, key=lambda p: p.as_posix().lower()):
        # 统一成字符串并规范化（去引号、反斜杠、..、url 编码等）
        rel_str = rel.as_posix() if hasattr(rel, "as_posix") else str(rel)
        rel_norm = Path(_normalize_relpath(rel_str))

        # 防御：拒绝绝对/越级/盘符；降级到 textures/<basename>
        if rel_norm.is_absolute() or any(part == ".." for part in rel_norm.parts) or re.match(r"^[A-Za-z]:", rel_str):
            rel_norm = Path("textures") / Path(rel_str).name

        dst = raw_dir / rel_norm
        if dst.exists() and not overwrite:
            skipped += 1
            # 既然已存在，也要记录到 placement_map
            placement_map[rel_norm.name] = rel_norm.as_posix()
            names.append(rel_norm.as_posix())
            continue

        # 本地查找：先按规范化相对路径，再按 basename 兜底
        src = _local_find_one(rel_norm, local_roots) or _local_find_one(Path(rel_norm.name), local_roots)
        if src:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, dst)   # 先硬链提速
                except Exception:
                    shutil.copy2(src, dst)
                ok += 1
                placement_map[rel_norm.name] = rel_norm.as_posix()
                names.append(rel_norm.as_posix())  # 记录相对路径！不是 name
            except Exception:
                fail += 1
        else:
            fail += 1

    # === 在复制完纹理后：把 raw 下的 MTL 重写为本地相对路径，并生成 *_fixed.obj ===
   
    #
    def _rewrite_mtl_maps_to_local(mtl_path: Path, raw_dir: Path, placement_map: dict[str, str]) -> tuple[Path, bool]:
        """
        仅对 raw_dir 下的 mtl 改写：
        - 把 C:\..., UNC, 绝对/越级 路径改为 textures/<basename> 或 placement_map 命中的相对路径
        - 返回 (fixed_mtl_path, changed), to fix the problem of using after autofix_obj_mtllib, in case 2 sets of fixed
        """
        DRIVE_RE = re.compile(r'^[A-Za-z]:[\\/]|^\\\\')
        lines = mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        changed = False

        for ln in lines:
            l = ln.lstrip().lower()
            if not (l.startswith("map_") or l.startswith("bump") or l.startswith("disp")):
                out.append(ln); continue

            parts = ln.split()
            if len(parts) < 2:
                out.append(ln); continue

            old = parts[-1]
            old_clean = str(old).strip().strip('"').strip("'").replace("\\", "/")
            bn = Path(old_clean).name

            # 先用我们刚放置的相对路径（names → placement_map）
            new_rel = placement_map.get(bn)

            # 兜底：raw 下按 basename 找
            if not new_rel:
                hits = list(raw_dir.rglob(bn))
                if hits:
                    try:
                        new_rel = hits[0].relative_to(raw_dir).as_posix()
                    except Exception:
                        new_rel = bn

            # 若是盘符/UNC/绝对/越级，强制落 textures/<bn>
            if not new_rel or DRIVE_RE.match(old_clean) or old_clean.startswith("/") or "/../" in f"/{old_clean}":
                new_rel = new_rel or f"textures/{bn}"

            if new_rel != old:
                changed = True
            parts[-1] = new_rel
            out.append(" ".join(parts))

        fixed = mtl_path.with_name(mtl_path.stem + "_fixed.mtl")
        if changed:
            fixed.write_text("\n".join(out), encoding="utf-8")
            return fixed, True
        else:
            # 无改动则不生成 *_fixed.mtl，复用原 mtl
            return mtl_path, False
    def _rewrite_obj_mtllibs(obj_path: Path, mtl_name_map: dict[str, str]) -> tuple[Path, bool]:
        """
        把 OBJ 中每条 mtllib 的文件名按映射替换。
        - 若 obj 已经是 *_fixed.obj：原地覆盖（不生成 *_fixed_fixed.obj）
        - 否则仅在发生改动时生成 *_fixed.obj
        返回 (最终 OBJ 路径, 是否发生改动)
        """
        lines = obj_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        out = []
        changed = False

        for ln in lines:
            if ln.lstrip().lower().startswith("mtllib"):
                parts = ln.split()
                if len(parts) >= 2:
                    old_name = Path(parts[-1]).name
                    new_name = mtl_name_map.get(old_name, old_name)
                    if new_name != parts[-1]:
                        changed = True
                    out.append(f"mtllib {new_name}")
                else:
                    out.append(ln)
            else:
                out.append(ln)

        is_fixed = obj_path.stem.endswith("_fixed")
        if changed:
            if is_fixed:
                # 覆盖现有 *_fixed.obj
                obj_path.write_text("\n".join(out), encoding="utf-8")
                return obj_path, True
            else:
                fixed_obj = obj_path.with_name(obj_path.stem + "_fixed.obj")
                fixed_obj.write_text("\n".join(out), encoding="utf-8")
                return fixed_obj, True
        else:
            # 无改动：保持原文件
            return obj_path, False

    # # 选用 raw 中的 mtl（否则用 obj 同目录），生成 *_fixed.mtl 与 *_fixed.obj
    # for m in mtls:
    #     mtl_in_raw = raw_dir / Path(m).name
    #     mtl_src = mtl_in_raw if mtl_in_raw.exists() 
    #     if not mtl_src.exists(): 
    #         continue
    #     fixed_mtl = _rewrite_mtl_maps_to_local(mtl_src, raw_dir, placement_map)
    #     try:
    #         lines = obj_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    #         out = []
    #         for ln in lines:
    #             if ln.lstrip().lower().startswith("mtllib"):
    #                 out.append(f"mtllib {fixed_mtl.name}")
    #             else:
    #                 out.append(ln)
    #         fixed_obj = obj_file.with_name(obj_file.stem + "_fixed.obj")
    #         fixed_obj.write_text("\n".join(out), encoding="utf-8")
    #         obj_file = fixed_obj  # 让上游后续落盘主 3D 就用 *_fixed.obj
    #     except Exception:
    #         pass
    mtl_name_map: dict[str, str] = {}  # 原 mtl 文件名 -> 最终使用的 mtl 文件名
    obj_mtls = parse_obj_for_mtllibs(obj_file) or []     # 可能已是 *_fixed.obj

    for m in obj_mtls:
        m_name = Path(m).name
        mtl_in_raw = raw_dir / m_name
        if not mtl_in_raw.exists():
            # 不再去 obj 父目录“猜测” mtl；缺就缺，交由统计报告
            continue

        final_mtl_path, _mtl_changed = _rewrite_mtl_maps_to_local(mtl_in_raw, raw_dir, placement_map)
        mtl_name_map[m_name] = final_mtl_path.name  # 仅替换文件名

    # 根据映射替换 OBJ 的 mtllib 行：已有 *_fixed.obj 则原地覆盖；否则有改动才生成 *_fixed.obj
    if mtl_name_map:
        new_obj_file, _obj_changed = _rewrite_obj_mtllibs(obj_file, mtl_name_map)
        obj_file = new_obj_file

    if local_only:
        # === 终态统计：expected/present/placed/missing 都按“相对路径”统计 ===
        expected = set()
        for m in mtls:
            loc = (raw_dir / Path(m).name)
            if not loc.exists():
                loc = (obj_file.parent / m)  # 这里 obj_file 已可能是 *_fixed.obj，不影响
            if loc.exists():
                for rel in parse_mtl_for_textures(loc):
                    rel_s = Path(_normalize_relpath((rel.as_posix() if hasattr(rel, "as_posix") else str(rel))))
                    if rel_s.is_absolute() or any(p == ".." for p in rel_s.parts):
                        rel_s = Path("textures") / rel_s.name
                    expected.add(rel_s.as_posix())

        present = set()
        for rel_s in expected:
            if (raw_dir / rel_s).exists():
                present.add(rel_s)

        placed = set(names)  # names 现在存的就是相对路径
        missing = sorted(expected - present - placed)

        print(f"    ▶ OBJ(local): mtl_copied={(raw_dir.glob('*.mtl')) and sum(1 for _ in raw_dir.glob('*.mtl')) or 0} "
            f"tex_copied={len(placed)} skipped={skipped} failed={fail} "
            f"(roots={', '.join(str(r) for r in local_roots)})")
        print(f"      · expected={len(expected)} present={len(present)} placed_now={len(placed)} missing={len(missing)}")
        if missing:
            print("      · TEX missing:", ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))

        # 返回保持你的签名，并附带额外报告信息
        report = {
            'expected_count': len(expected),
            'expected_names': sorted(expected),
            'present_names': sorted(present),
            'missing': missing,
            'repo_missing': [],
            'missing_reason_map': {m: 'local_missing' for m in missing},
        }
        return (ok, fail, skipped, names, report)
        # # 2) 打印概要
        # print(f"    ▶ OBJ(local): mtl_copied={len(placed)} tex_copied={len(placed_tex)} "
        #       f"skipped={skipped} failed={fail} (roots={', '.join(str(r) for r in local_roots)})")
        # if placed:
        #     print("      · MTL placed: " + ", ".join(placed[:10]) + (" ..." if len(placed) > 10 else ""))
        # if placed_tex:
        #     print("      · TEX placed: " + ", ".join(placed_tex[:10]) + (" ..." if len(placed_tex) > 10 else ""))
        # if missing_tex:
        #     print("      · TEX missing: " + ", ".join(missing_tex[:10]) + (" ..." if len(missing_tex) > 10 else ""))

        # # 3) names 已包含成功拷贝到 raw/ 的文件名；返回计数保持一致
        # return (ok, fail, skipped, names)

    # 2) 汇总所有 MTL 内的纹理相对路径
    tex_refs: set[Path] = set()
    for m in mtls:
        mtl_local = (obj_file.parent / m).resolve()
        if not mtl_local.exists():
            # 若本地没有 mtl，先尝试把它下载下来（和 OBJ 同上游目录）
            try:
                base_url = re.sub(r"[^/]+$", "", src_url)
                mtl_url = _join_url_dir_inline(base_url, _normalize_relpath(str(m)))
                dst_mtl = raw_dir / m
                dst_mtl.parent.mkdir(parents=True, exist_ok=True)
                safe_download(mtl_url, dst_mtl, overwrite=overwrite)
                mtl_local = dst_mtl
                print(f"    · fetched missing MTL: {m}")
            except Exception as e:
                print(f"    ⚠ missing MTL, and fetch failed: {m} ({e})")
                continue
        tex_refs |= parse_mtl_for_textures(mtl_local)

    if not tex_refs:
        print("    · no texture refs in MTL (or MTL missing)")
        return (0, 0, 0, [])

    # 3) 并发拉取每一张贴图（直链→仓库兜底）
    base_url = re.sub(r"[^/]+$", "", src_url)
    base_url = fix_github_blob(base_url)
    gh = _github_parse_url_inline(base_url)  # (user, repo, sha, dir_path, filename) or None

    def _fetch_one(rel_in: Path):
        """返回 (status, rel_norm_str, src_used)
           status: 'ok' | 'skip' | 'fail'
        """
        rel_norm = Path(_normalize_relpath(rel_in.as_posix()))
        rel_str  = rel_norm.as_posix()
        def _looks_win_abs(s: str) -> bool:
            return bool(re.match(r"^[a-zA-Z]:[\\/]", s)) or s.startswith("\\\\") or s.startswith("//")

        # Windows 绝对/UNC 路径 → 只保留文件名
        safe_rel = Path(rel_norm.name) if _looks_win_abs(rel_str) else rel_norm
        # dst = (raw_dir / rel_norm)
        dst = raw_dir / safe_rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 已存在：跳过
        if dst.exists() and not overwrite:
            return ("skip", rel_norm.as_posix(), "(local)")

        # 先直链
        primary_url = _join_url_dir_inline(base_url, rel_norm.as_posix())
        try:
            safe_download(primary_url, dst, overwrite=overwrite)
            return ("ok", rel_norm.as_posix(), primary_url)
        except Exception:
            pass  # 进入兜底

        # GitHub 仓库兜底（按文件名）
        if gh:
            user, repo, sha, dir_path, _ = gh
            fname = Path(rel_norm.name).name

            # === NEW: 生成候选名，原名优先，其次才是 _/-/空格 互换 ===
            for cand in filename_variants_exact_first(fname):
                # 先在“就近目录 / 常见子目录”搜
                raw_urls = _github_search_texture_in_repo_inline(
                    user, repo, sha, dir_path, cand, subdirs=SUBDIR_CANDIDATES_INLINE
                )
                # 若未命中，再做浅递归兜底
                if not raw_urls:
                    for rp in _github_list_files_recursive_inline(user, repo, sha, dir_path, max_depth=GITHUB_ENUM_MAX_DEPTH):
                        if Path(rp).name.lower() == cand.lower():
                            raw_urls.append(_github_raw_url_inline(user, repo, sha, rp))
                            break

                # 命中就下载并返回
                for alt in raw_urls:
                    try:
                        safe_download(alt, dst, overwrite=overwrite)
                        return ("ok", rel_norm.as_posix(), alt)
                    except Exception:
                        continue
        # 最终失败
        return ("fail", rel_norm.as_posix(), "")

    tasks = []
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        for rel in sorted(tex_refs, key=lambda p: p.as_posix().lower()):
            tasks.append(ex.submit(_fetch_one, rel))

        results = [fu.result() for fu in tasks]

    fetched = []
    for status, rel_str, src in results:
        if status == "ok":
            ok += 1; fetched.append(rel_str)
        elif status == "skip":
            skipped += 1
        else:
            fail += 1

    if fetched:
        names.extend(fetched)

    print(f"    ▶ OBJ textures: fetched={ok}, skipped={skipped}, failures={fail}")
    if names:
        print("      ↳ " + ", ".join(names))

    return (ok, fail, skipped, names)


# === GLB external refs ===
# def ensure_complete_glb_asset(glb_file: Path, src_url: str, raw_dir: Path, overwrite=False):
#     """
#     GLB 外联资源补全（若有的话）：
#     - 解析 GLB JSON 块，读取 images/buffers 里的 uri
#     - 对于非 data: 的 uri（相对路径或 http(s)），尝试从 src_url 所在目录或仓库中下载到本地（与 GLB 同目录）
#     - 做有效性校验；统计 downloaded/missing/invalid
#     - 打印类似 FBX 的汇总日志
#     """
#     base_url = re.sub(r"[^/]+$", "", fix_github_blob(src_url))
#     glb_dir = glb_file.parent
#     externals = glb_referenced_externals(glb_file)

#     if not externals:
#         print("    ▶ GLB externals summary: refs=0 (all embedded); nothing to fetch.")
#         return {
#             'kind': 'glb',
#             'refs_total': 0,
#             'refs_names': [],
#             'downloaded': 0,
#             'invalid_downloaded': 0,
#             'missing_names': [],
#             'found_but_failed': [],
#             'exported_embedded': 0,
#             'tried_count': 0,
#             'fetched_names': [],
#             'fetched_invalid_names': [],
#             'fetched_sources': [],
#             'tex_index_check': 'n/a',
#             'tex_index_blender_count': 0,
#             'tex_index_regex_count': 0,
#         }

#     downloaded_ok = 0; fetched_ok = []; fetched_invalid = []; missing = []; sources = []
#     repo_found_but_failed = []; tried = 0; invalid_reasons = {}

#     for rel in externals:
#         tried += 1
#         # 目标落在 glb 同目录（保持相对层级）
#         dst = (glb_dir / rel).resolve()
#         dst.parent.mkdir(parents=True, exist_ok=True)

#         # 1) 先尝试“同目录拼接”
#         cand_url = _join_url_dir_inline(base_url, rel.as_posix())
#         ok, reason = _check_download_reason_inline(rel.name)
#         if ok:
#             try:
#                 u = fix_github_blob(cand_url)
#                 safe_download(u, dst, overwrite=overwrite)
#                 downloaded_ok += 1; fetched_ok.append(str(rel)); sources.append(u)
#                 continue
#             except Exception as e:
#                 invalid_reasons[str(rel)] = f"http_error:{e}"

#         # 2) 若是 GitHub，仓库内按文件名兜底搜（含递归枚举等你已实现的逻辑）
#         hits = _github_search_texture_in_repo_inline(src_url, rel.name)
#         if hits:
#             # 逐个试下载（直到一个成功）
#             success = False
#             for u in hits:
#                 try:
#                     safe_download(u, dst, overwrite=overwrite)
#                     downloaded_ok += 1; fetched_ok.append(str(rel)); sources.append(u)
#                     success = True
#                     break
#                 except Exception as e:
#                     invalid_reasons[str(rel)] = f"http_error:{e}"
#             if not success:
#                 repo_found_but_failed.append(str(rel))
#                 continue
#         else:
#             # 3) 如果 rel 看起来本身就是 http(s) 绝对 URL，也尝试直接拉本地化
#             if re.match(r"^https?://", rel.as_posix(), re.I):
#                 try:
#                     safe_download(rel.as_posix(), dst, overwrite=overwrite)
#                     downloaded_ok += 1; fetched_ok.append(str(rel)); sources.append(rel.as_posix())
#                     continue
#                 except Exception as e:
#                     invalid_reasons[str(rel)] = f"http_error:{e}"
#             missing.append(str(rel))

#     print(f"    ▶ GLB externals summary: refs={len(externals)}, downloaded_ok={downloaded_ok}, "
#           f"invalid={len(fetched_invalid)}, missing={len(missing)}")
#     if missing:
#         print("      · verified missing in repo/urls:", ", ".join(missing[:10]) + (" ." if len(missing)>10 else ""))
#     if repo_found_but_failed:
#         print("      · found in repo but failed to fetch/validate:", ", ".join(repo_found_but_failed[:10]) + (" ." if len(repo_found_but_failed)>10 else ""))
#     if invalid_reasons:
#         print("      · attempt failures:", ", ".join(f"{k}:{v}" for k,v in invalid_reasons.items()))

#     return {
#         'kind': 'glb',
#         'refs_total': len(externals),
#         'refs_names': [p.as_posix() for p in externals],
#         'downloaded': downloaded_ok,
#         'invalid_downloaded': len(fetched_invalid),
#         'missing_names': missing,
#         'found_but_failed': repo_found_but_failed,
#         'exported_embedded': 0,
#         'tried_count': tried,
#         'fetched_names': fetched_ok,
#         'fetched_invalid_names': fetched_invalid,
#         'fetched_sources': sources,
#         'tex_index_check': 'n/a',
#         'tex_index_blender_count': 0,
#         'tex_index_regex_count': 0,
#     }
def ensure_complete_glb_asset(glb_file: Path, src_url: str, raw_dir: Path, overwrite=False,
                              local_only: bool=False, local_roots: list[Path] | None=None):
    """
    GLB 外联资源补全（并发 + 正确的直链优先）：
    - 从 GLB JSON 中读出 images/buffers 的非 data: URI
    - 先尝试“与 GLB 同目录”的直链；失败再用 GitHub 仓库搜索兜底；最后支持 http(s) 绝对 URI 的本地化
    - 并发执行；最终打印一次汇总
    """
    local_roots = [p for p in (local_roots or []) if isinstance(p, Path) and p.exists()]
    refs = glb_referenced_externals(glb_file)
    base_url = re.sub(r"[^/]+$", "", fix_github_blob(src_url))
    glb_dir = glb_file.parent
    externals = glb_referenced_externals(glb_file)

    if not externals:
        print("    ▶ GLB externals summary: refs=0 (all embedded); nothing to fetch.")
        return {
            'kind': 'glb',
            'refs_total': 0,
            'refs_names': [],
            'downloaded': 0,
            'invalid_downloaded': 0,
            'missing_names': [],
            'repo_missing_names': [],
            'missing_reason_map': {},
            'found_but_failed': [],
            'exported_embedded': 0,
            'tried_count': 0,
            'fetched_names': [],
            'fetched_invalid_names': [],
            'fetched_sources': [],
            'tex_index_check': 'n/a',
            'tex_index_blender_count': 0,
            'tex_index_regex_count': 0,
        }
    ok = fail = 0; names = []
    missing_local: list[str] = []
    for rel in refs:
        relp = Path(_normalize_relpath(rel.as_posix()))
        dst = raw_dir / relp
        if dst.exists() and not overwrite:
            continue
        src = _local_find_one(rel, local_roots) or _local_find_one(Path(rel.name), local_roots)
        if src:
            try:
                shutil.copy2(src, dst); ok += 1; names.append(dst.name)
            except Exception:
                fail += 1
        else:
            missing_local.append(relp.as_posix())
    if local_only:
        return {'kind':'glb','refs_total': len(refs),'downloaded': ok,'missing_names': missing_local,
                'repo_missing_names': [], 'missing_reason_map': {m: 'local_missing' for m in missing_local},
                'exported_embedded': 0,'fetched_names': names,'invalid_downloaded': 0,
                'fetched_invalid_names': [],'fetched_sources': [],'tex_index_check': 'local_only',
                'tex_index_blender_count': 0,'tex_index_regex_count': 0}
    
    gh = _github_parse_url_inline(src_url) or _github_parse_url_inline(base_url)
    repo_missing: list[str] = []
    missing_reason_map: dict[str, str] = {}

    def _fetch_one_glb(rel_in: Path):
        """返回 (status, rel_str, src_used, repo_missing_flag, reason_tag)。"""
        rel = Path(_norm_relpath(rel_in.as_posix()))
        dst = (glb_dir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 已存在：跳过
        if dst.exists() and not overwrite:
            return ("skip", rel.as_posix(), "(local)", False, "local")

        # 1) 直链（与 GLB 同目录）
        primary = _join_url_dir_inline(base_url, rel.as_posix())
        try:
            safe_download(primary, dst, overwrite=overwrite)
            return ("ok", rel.as_posix(), primary, False, "direct")
        except Exception:
            pass

        # 2) GitHub 仓库兜底（按文件名）
        repo_missing_flag = False
        if gh:
            user, repo, sha, dir_path, _ = gh
            fname = Path(rel.name).name
            raw_urls = _github_search_texture_in_repo_inline(
                user, repo, sha, dir_path, fname, subdirs=SUBDIR_CANDIDATES_INLINE
            )
            if not raw_urls:
                repo_missing_flag = True
                for rp in _github_list_files_recursive_inline(user, repo, sha, dir_path, max_depth=GITHUB_ENUM_MAX_DEPTH):
                    if Path(rp).name.lower() == fname.lower():
                        raw_urls.append(_github_raw_url_inline(user, repo, sha, rp))
                        break
                if raw_urls:
                    repo_missing_flag = False
            for alt in raw_urls:
                try:
                    safe_download(alt, dst, overwrite=overwrite)
                    return ("ok", rel.as_posix(), alt, False, "repo")
                except Exception:
                    continue

        # 3) 绝对 URL（http/https）本地化
        if re.match(r"^https?://", rel.as_posix(), re.I):
            try:
                safe_download(rel.as_posix(), dst, overwrite=overwrite)
                return ("ok", rel.as_posix(), rel.as_posix(), False, "http")
            except Exception:
                pass

        reason_tag = "repo_missing" if repo_missing_flag else "exhausted"
        return ("fail", rel.as_posix(), "", repo_missing_flag, reason_tag)

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        futs = [ex.submit(_fetch_one_glb, rel) for rel in externals]
        results = [fu.result() for fu in futs]

    downloaded_ok = 0; skipped = 0; missing = []; fetched_ok = []; sources = []
    for status, rel_str, src, repo_missing_flag, reason_tag in results:
        if status == "ok":
            downloaded_ok += 1; fetched_ok.append(rel_str); sources.append(src or "(guessed)")
        elif status == "skip":
            skipped += 1
        else:
            missing.append(rel_str)
            if repo_missing_flag and rel_str not in repo_missing:
                repo_missing.append(rel_str)
            if reason_tag:
                missing_reason_map[rel_str] = reason_tag

    for rel_str in missing:
        missing_reason_map.setdefault(rel_str, "unknown")

    print(f"    ▶ GLB externals summary: refs={len(externals)}, downloaded_ok={downloaded_ok}, skipped={skipped}, missing={len(missing)}")
    if repo_missing:
        print("      · repo missing: " + ", ".join(repo_missing[:10]) + (" ..." if len(repo_missing)>10 else ""))
    remaining_missing = sorted(set(missing) - set(repo_missing))
    if remaining_missing:
        print("      · unresolved missing: " + ", ".join(remaining_missing[:10]) + (" ..." if len(remaining_missing)>10 else ""))

    return {
        'kind': 'glb',
        'refs_total': len(externals),
        'refs_names': [p.as_posix() for p in externals],
        'downloaded': downloaded_ok,
        'invalid_downloaded': 0,
        'missing_names': missing,
        'repo_missing_names': sorted(repo_missing),
        'missing_reason_map': missing_reason_map,
        'found_but_failed': [],
        'exported_embedded': 0,
        'tried_count': len(externals),
        'fetched_names': fetched_ok,
        'fetched_invalid_names': [],
        'fetched_sources': sources,
        'tex_index_check': 'n/a',
        'tex_index_blender_count': 0,
        'tex_index_regex_count': 0,
    }

def _parse_glb_json_inline(glb_path: Path) -> dict:
    import struct, json
    with open(glb_path, 'rb') as f:
        header = f.read(12)
        if len(header) < 12 or header[:4] != b'glTF':
            return {}
        length = struct.unpack('<I', header[8:12])[0]
        # 读第一个 chunk
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            return {}
        chunk_len, chunk_type = struct.unpack('<II', chunk_header)
        if chunk_type != 0x4E4F534A:  # 'JSON'
            return {}
        json_bytes = f.read(chunk_len)
    try:
        return json.loads(json_bytes.decode('utf-8', errors='ignore'))
    except Exception:
        return {}

def glb_referenced_externals(glb_path: Path) -> list[Path]:
    """
    返回 GLB 中 images/buffers 里出现的非 data: 且非 http(s) 的 URI（相对路径）。
    """
    J = _parse_glb_json_inline(glb_path)
    refs: set[Path] = set()
    for key in ("images", "buffers"):
        for item in (J.get(key) or []):
            uri = (item or {}).get("uri")
            if not uri or isinstance(uri, dict):
                continue
            u = str(uri).strip()
            if u.lower().startswith("data:"):
                continue
            if re.match(r"^https?://", u, flags=re.I):
                # 也支持把绝对 URL 拉到本地，以便离线
                refs.add(Path(_norm_relpath(u)))
            else:
                refs.add(Path(_norm_relpath(u)))
    return sorted(refs, key=lambda p: p.as_posix().lower())


############# fbx 解析相关 ################

import re
from urllib.parse import urljoin

FBX_TEX_EXTS = (".png",".jpg",".jpeg",".tga",".tiff",".bmp",".exr",".dds",".ktx2")

def _norm_relpath(s: str) -> str:
    s = s.strip().strip('"').strip("'").replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    return s

def fbx_referenced_textures(fbx_path: Path) -> list[Path]:
    """
    超轻量：直接从二进制里找出以常见图片后缀结尾的路径片段。
    返回相对路径列表（尽量保留如 'textures/diffuse.png' 或文件名）。
    """
    refs = set()
    data = fbx_path.read_bytes()
    for m in re.finditer(rb'[^ \t\r\n\x00"]+?\.(png|jpg|jpeg|tga|tiff|bmp|exr|dds|ktx2)', data, re.IGNORECASE):
        try:
            s = m.group(0).decode("utf-8", errors="ignore")
        except Exception:
            continue
        s = s.replace("\\", "/")
        # 取最后 1~2 段，避免绝对盘符/长 URL 污染
        parts = [p for p in s.split("/") if p]
        if not parts:
            continue
        tail = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        refs.add(Path(_norm_relpath(tail)))
    # 去掉疑似 URL（http/https）——我们只想保留相对路径
    refs = {p for p in refs if not re.match(r"^https?://", p.as_posix(), re.I)}
    return sorted(refs, key=lambda p: p.as_posix().lower())

def ensure_complete_fbx_asset(fbx_file: Path, src_url: str, raw_dir: Path, overwrite=False, 
                              local_only: bool=False, local_roots: list[Path] | None=None):
    """
    FBX 贴图完整化（报告化）：
    - Blender（若可用）列引用 & 导出内嵌贴图到 <raw>/<fbx_stem>.fbm/
    - 外链贴图：先基于 src_url 同目录候选路径尝试；如是 GitHub，再在仓库内按文件名就近搜索并下载
    - 对下载文件做有效性校验；记录 invalid / not_found_in_repo
    - 与正则启发式索引交叉比对，报告一致性
    返回 dict：见末尾
    """
    import os, shutil
    def _is_placeholder_fbx_name(n: str) -> bool:
        """
        识别 FBX/Blender 导入产生的“占位名”，如 'Map #1' / 'Map #1.png' 等：
        - 无路径、无扩展名或
        - 名字形如 'Map #<num>' / 'Tex #<num>'（可带扩展名）
        """
        if not n: 
            return True
        s = str(n).strip().strip('"').strip("'")
        bn = Path(s).name  # 仅基名
        if not Path(bn).suffix:  # 没有扩展名，基本是占位
            return True
        if re.match(r'^(map|tex)\s*#\d+(\.\w+)?$', bn, flags=re.I):
            return True
        return False

# （可选）如果你不想下载 PSD：把 PSD 排除掉（保留其它格式）
    FBX_TEX_EXTS = tuple({'.png','.jpg','.jpeg','.tga','.bmp','.tiff','.ktx2','.dds'})  # 无 .psd
    def _ext_allowed(n: str) -> bool:
        return Path(str(n)).suffix.lower() in FBX_TEX_EXTS
    base_url = re.sub(r"[^/]+$", "", fix_github_blob(src_url))

    # fbm_dir = fbx_file.with_suffix('.fbm') if TEX_OUT_MODE == "fbm" else fbx_file.parent
    # fbm_dir.mkdir(parents=True, exist_ok=True)
    local_roots = [p for p in (local_roots or []) if isinstance(p, Path) and p.exists()]
    fbm_dir = raw_dir / (fbx_file.stem + ".fbm")
    fbm_dir.mkdir(parents=True, exist_ok=True) # for trellis download
    # 记录本轮运行开始前，fbm_dir 中已存在的文件（只存 basename）
    pre_existing_basenames = set()
    try:
        if fbm_dir.exists():
            for _p in fbm_dir.iterdir():
                if _p.is_file():
                    pre_existing_basenames.add(_p.name)
    except Exception:
        pass
    if overwrite :
        shutil.rmtree(fbm_dir, ignore_errors=True)
        fbm_dir.mkdir(parents=True, exist_ok=True)
        pre_existing_basenames = set()  # 刚清空，当然没有本地旧文件    
    # A) Blender 引用 + 内嵌导出
    refs_via_blender, exported = _blender_list_and_export_images(fbx_file, fbm_dir)
    exported_set = set(exported)

    # B) 正则兜底（从二进制扒路径尾巴/文件名）
    def _regex_refs(p: Path) -> set[str]:
        refs = set()
        try:
            blob = p.read_bytes()
        except Exception:
            return refs
        for m in re.finditer(rb'[^ \t\r\n\x00"]+?\.(png|jpg|jpeg|tga|tif|tiff|bmp|exr|dds|webp|ktx2?)', blob, re.IGNORECASE):
            s = m.group(0).decode('utf-8', errors='ignore').replace('\\','/')
            parts = [seg for seg in s.split('/') if seg]
            tail = '/'.join(parts[-2:]) if parts else ''
            if tail: refs.add(tail)
        return refs

    # 归一化“关键文件名集”
    blender_names = set()
    for nm in refs_via_blender:
        nm = (nm or "").strip().replace("\\","/")
        if nm:
            blender_names.add(nm.split("/")[-1])
    regex_names = {Path(s).name for s in _regex_refs(fbx_file)}
    # all_names = sorted(set(blender_names) | set(regex_names))
    # 你现有：blender_names, regex_names 这两个集合/列表
    # 先剔除占位名（避免反复 http 尝试 'Map #1.png'）
    blender_placeholders = {x for x in blender_names if _is_placeholder_fbx_name(x)}
    regex_placeholders   = {x for x in regex_names   if _is_placeholder_fbx_name(x)}
    if blender_placeholders:
        print("      · skip placeholders (no file path): " + ", ".join(list(blender_placeholders)[:5]) + (" ..." if len(blender_placeholders) > 5 else ""))
    blender_names = [x for x in blender_names if x not in blender_placeholders]
    # regex_names   = [x for x in regex_placeholders if x not in regex_placeholders]  # 也可直接置空/忽略
    regex_names = [x for x in regex_names if x not in regex_placeholders]

    # 按你原本规则合并后，再按扩展做一次过滤（例如排除 .psd）
    all_names_raw = sorted(set(blender_names) | set(regex_names))
    all_names = [n for n in all_names_raw if _ext_allowed(n)]

    # 索引一致性
    def _norm(n:str):
        return n.strip().strip('"').strip("'").replace("\\","/").lower()
    bset = {_norm(x) for x in blender_names}
    rset = {_norm(x) for x in regex_names}
    only_blender = sorted(bset - rset)
    only_regex   = sorted(rset - bset)
    # index_check = "ok" if (len(only_blender)==0 and len(only_regex)==0) else \
    #               f"mismatch: blender_only={only_blender[:5]} regex_only={only_regex[:5]}"

    # C) 下载策略（并发化 + 过滤无效占位名）
    SUBDIRS = ('Textures','textures','Materials','materials','images','Images','maps','Maps', 'Materials and Textures')
    tried = 0; downloaded_ok = 0; downloaded_invalid = 0; satisfied_local = 0
    fetched_ok = []; fetched_invalid = []; missing = []; sources = []
    invalid_reasons = {}
    repo_found_but_failed = []
    repo_missing = []
    missing_reason_map: dict[str, str] = {}

    gh = _github_parse_url_inline(src_url)

    # 过滤掉明显“占位/无扩展名”的条目，避免浪费请求（例如 'Map #1'）
    def _looks_like_texname(n: str) -> bool:
        n2 = (n or "").strip().strip('"').strip("'")
        if not n2:
            return False
        bn = Path(n2).name
        if not any(bn.lower().endswith(ext) for ext in FBX_TEX_EXTS):
            return False
        # 过滤无扩展或奇怪占位符
        if bn.lower().startswith("map #") and "." not in bn:
            return False
        return True

    wanted_names = [name for name in all_names if _looks_like_texname(name)]
    satisfied_local = 0
    placed_basenames = set()  # FIX: 收集已放置的 basename，供 local_only 统计
    for rel in wanted_names:
        src = _local_find_one(Path(rel), local_roots) or _local_find_one(Path(Path(rel).name), local_roots)
        if src:
            dst = fbm_dir / Path(rel).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or overwrite:
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
            if dst.exists():
                satisfied_local += 1
                placed_basenames.add(dst.name)  # FIX: 记录成功放置的文件名

    # --- local_only 早退（FIX: 避免把字符串当 Path 用） ---
    if local_only:
        # 以 basename 对应回原始 wanted_names，区分“已放置/缺失”
        wanted_base = [Path(n).name for n in wanted_names]
        fetched_names = sorted({bn for bn in wanted_base if (fbm_dir / bn).exists()})
        # missing 保留原始名字（含相对路径片段），便于排查
        missing_names = [orig for orig, bn in zip(wanted_names, wanted_base) if (fbm_dir / bn).exists() is False]
        return {
            'kind': 'fbx',
            'refs_total': len(wanted_names),
            'downloaded': 0,
            'invalid_downloaded': 0,
            'missing_names': missing_names,         # FIX
            'found_but_failed': [],
            'repo_missing_names': [],
            'missing_reason_map': {},
            'exported_embedded': len(exported_set),
            'tried_count': 0,
            'fetched_names': fetched_names,         # FIX
            'fetched_invalid_names': [],
            'fetched_sources': [],
            'tex_index_check': 'local_only',
            'tex_index_blender_count': len(blender_names),
            'tex_index_regex_count': len(regex_names),  # FIX: 统计 regex 索引数，而非 wanted
            'satisfied_local': satisfied_local,     # 可选：便于调试
        }
    def _try_fetch_one(name: str) -> tuple[str, str, bool, bool, bool, str]:
        """返回 (name, source, ok, invalid, repo_missing, reason_tag)。"""
        nonlocal tried
        # --- 1) 直连：同目录/约定子目录（含 <fbx>.fbm） ---
        candidates = [name, f"{fbx_file.stem}.fbm/{name}"] + [f"{sd}/{name}" for sd in SUBDIRS]
        for rel in candidates:
            tried += 1
            url = _join_url_dir_inline(base_url, rel)
            dst = fbm_dir / name
            if dst.exists():
                ok, _reason = _check_download_reason_inline(dst, expect_ext=Path(name).suffix)
                if ok:
                    if FBX_LOG_ORIGIN:
                        print(f"      · skip download (already satisfied): {name}")
                    return (name, "(local)", True, False, False, "local")
                else:
                    try: dst.unlink()
                    except Exception: pass
            try:
                safe_download(url, dst, overwrite=overwrite)
                ok, reason = _check_download_reason_inline(dst, expect_ext=Path(name).suffix)
                if ok:
                    return (name, url, True, False, False, "direct")
                else:
                    try: dst.unlink(missing_ok=True)
                    except Exception: pass
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            except Exception:
                invalid_reasons["http_error"] = invalid_reasons.get("http_error", 0) + 1

        # --- 2) GitHub 仓库内搜索（Tree 缓存优先，再就近 + 递归兜底） ---
        repo_missing_flag = False
        if gh:
            user, repo, sha, dir_path, _ = gh
            # 先试就近 + 递归（内部已优先用 Tree 缓存）
            raw_urls = _github_search_texture_in_repo_inline(user, repo, sha, dir_path, name, SUBDIRS) or []
            if not raw_urls:
                repo_missing_flag = True

            # UDIM/正则：如果文件名里含 UDIM 语义，再用 Tree 缓存正则匹配
            low = name.lower()
            if ("<udim>" in low) and not raw_urls:
                # <UDIM> → 正则 (\d{4})
                bn = Path(name).name
                pattern = "^" + re.escape(bn).replace("\\<udim\\>", "(\\d{4})") + "$"
                raw_urls = _github_api_find_raw_urls_by_regex_inline(user, repo, sha, pattern, limit=64)
                if raw_urls:
                    repo_missing_flag = False

            # 逐个候选尝试
            for url in raw_urls:
                try:
                    dst = fbm_dir / name
                    safe_download(url, dst, overwrite=overwrite)
                    ok, reason = _check_download_reason_inline(dst, expect_ext=Path(name).suffix)
                    if ok:
                        return (name, url, True, False, False, "repo")
                    else:
                        try: dst.unlink(missing_ok=True)
                        except Exception: pass
                        invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                        if name not in repo_found_but_failed:
                            repo_found_but_failed.append(name)
                except Exception:
                    invalid_reasons["http_error"] = invalid_reasons.get("http_error", 0) + 1
                    if name not in repo_found_but_failed:
                        repo_found_but_failed.append(name)

        # 没有成功
        reason_tag = "repo_missing" if repo_missing_flag else "exhausted"
        return (name, "", False, False, repo_missing_flag, reason_tag)

    # --- 并发抓取 ---
    from concurrent.futures import ThreadPoolExecutor, as_completed
    futures = []
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as ex:
        for nm in wanted_names:
            futures.append(ex.submit(_try_fetch_one, nm))
        for fut in as_completed(futures):
            name, source, ok, invalid, repo_missing_flag, reason_tag = fut.result()
            if ok:
                downloaded_ok += 1
                fetched_ok.append(name)
                sources.append(source)
            else:
                # 没抓到就记 missing；（invalid 在上面已计入 invalid_reasons 并删除了文件）
                missing.append(name)
                if repo_missing_flag and name not in repo_missing:
                    repo_missing.append(name)
                if reason_tag:
                    missing_reason_map[name] = reason_tag

    for name in missing:
        missing_reason_map.setdefault(name, "unknown")

    # —— 统计 & 日志（保持你原先的字段/口径）——
    # exported_only = sorted(set(exported_set) - set(fetched_ok))
    if not only_blender and not only_regex:
        index_check = "ok"
    # elif missing == 0:
    #     index_check = f"ok (names differ; all satisfied; blender_only={only_blender[:5]} regex_only={only_regex[:5]})"
    # else:
    #     index_check = f"mismatch: blender_only={only_blender[:5]} regex_only={only_regex[:5]}"
    index_check = "ok" if (not only_blender and not only_regex) else \
              (f"ok (names differ; all satisfied; blender_only={only_blender[:5]} regex_only={only_regex[:5]})"
               if missing == 0 else
               f"mismatch: blender_only={only_blender[:5]} regex_only={only_regex[:5]}")
    
    sat_local = locals().get("satisfied_local", 0)
    embedded_count = len(exported_set) if 'exported_set' in locals() else 0
    print(f"    ▶ FBX textures summary: refs={len(all_names)}, "
        f"downloaded_ok={downloaded_ok}, embedded_exported={embedded_count}, "
        f"satisfied_local={sat_local}, invalid={downloaded_invalid}, missing={len(missing)}")
    if repo_missing:
        print("      · repo missing:", ", ".join(repo_missing[:10]) + (" ..." if len(repo_missing)>10 else ""))
    unresolved_missing = sorted(set(missing) - set(repo_missing) - set(repo_found_but_failed))
    if unresolved_missing:
        print("      · unresolved missing:", ", ".join(unresolved_missing[:10]) + (" ..." if len(unresolved_missing)>10 else ""))
    if repo_found_but_failed:
        print("      · found in repo but failed to fetch/validate:", ", ".join(repo_found_but_failed[:10]) + (" ..." if len(repo_found_but_failed)>10 else ""))
    if fetched_invalid:
        print("      · invalid files:", ", ".join(fetched_invalid[:10]) + (" ..." if len(fetched_invalid)>10 else ""))
    # if invalid_reasons:
    #     print("      · invalid reasons:", ", ".join(f"{k}:{v}" for k,v in invalid_reasons.items()))
    if (downloaded_invalid > 0 or missing or repo_found_but_failed or fetched_invalid) and invalid_reasons:
        print("      · attempt failures:", ", ".join(f"{k}:{v}" for k,v in invalid_reasons.items()))
    if index_check != "ok":
        print("      · index check:", index_check)

    return {
        'kind': 'fbx',
        'refs_total': len(wanted_names),
        'downloaded': downloaded_ok,
        'invalid_downloaded': sum(invalid_reasons.values()),
        'missing_names': sorted(set(missing) - set(fetched_ok)),
        'found_but_failed': repo_found_but_failed,
        'repo_missing_names': sorted(repo_missing),
        'missing_reason_map': missing_reason_map,
        'exported_embedded': len(exported_set),
        'tried_count': tried,
        'fetched_names': fetched_ok,
        'fetched_invalid_names': [k for k, v in invalid_reasons.items() for _ in range(v)],
        'fetched_sources': sources,
        'tex_index_check': index_check,
        'tex_index_blender_count': len(blender_names),
        'tex_index_regex_count': len(regex_names),
    }

def ensure_complete_asset_anyformat(asset_path: Path, src_url: str, raw_dir: Path, overwrite=False,
                                    local_only: bool=False,
                                    local_roots: list[Path] | None=None):
    """按扩展名补齐贴图/材质并返回统一报告 dict。"""
    ext = asset_path.suffix.lower()
    if ext == '.obj':
        try:
            result = ensure_complete_obj_asset_strict(
                asset_path, src_url, raw_dir, overwrite=overwrite,
                local_only=local_only, local_roots=local_roots or []
            )
            extra: dict[str, Any] = {}
            if isinstance(result, tuple) and len(result) >= 5:
                dn, fails, skipped, names, extra = result[:5]
            else:
                dn, fails, skipped, names = result
            missing_names = list(extra.get('missing', [])) if isinstance(extra, dict) else []
            repo_missing = list(extra.get('repo_missing', [])) if isinstance(extra, dict) else []
            missing_reason_map = dict(extra.get('missing_reason_map', {})) if isinstance(extra, dict) else {}
            if not missing_reason_map and missing_names:
                missing_reason_map = {m: 'local_missing' for m in missing_names}
            refs_total = extra.get('expected_count') if isinstance(extra, dict) else None
            if refs_total is None:
                refs_total = dn + fails + len(missing_names)
            return {'kind':'obj','refs_total': refs_total, 'downloaded': dn, 'missing_names': missing_names,
                    'exported_embedded': 0, 'fetched_names': names,
                    'invalid_downloaded': 0, 'fetched_invalid_names': [], 'fetched_sources': [], 'tex_index_check': 'n/a',
                    'tex_index_blender_count': 0, 'tex_index_regex_count': 0,
                    'repo_missing_names': repo_missing, 'missing_reason_map': missing_reason_map}
        except Exception as e:
            print(f'    ⚠ OBJ texture supplement skipped: {e}')
            return {'kind':'obj','refs_total': 0, 'downloaded': 0, 'missing_names': [], 'exported_embedded': 0, 'fetched_names': [],
                    'invalid_downloaded': 0, 'fetched_invalid_names': [], 'fetched_sources': [], 'tex_index_check': 'n/a',
                    'tex_index_blender_count': 0, 'tex_index_regex_count': 0,
                    'repo_missing_names': [], 'missing_reason_map': {}}
    elif ext == '.fbx':
        return ensure_complete_fbx_asset(asset_path, src_url, raw_dir, overwrite=overwrite,
                                         local_only=local_only, local_roots=local_roots or [])
    elif ext == '.glb':
        return ensure_complete_glb_asset(asset_path, src_url, raw_dir, overwrite=overwrite,
                                         local_only=local_only, local_roots=local_roots or [])
    else:
        return {'kind':'other','refs_total': 0, 'downloaded': 0, 'missing_names': [], 'exported_embedded': 0, 'fetched_names': [],
                'invalid_downloaded': 0, 'fetched_invalid_names': [], 'fetched_sources': [], 'tex_index_check': 'n/a',
                'tex_index_blender_count': 0, 'tex_index_regex_count': 0,
                'repo_missing_names': [], 'missing_reason_map': {}}

# def _bulk_fetch_parent_selective_inline(mesh_path: Path, source_url: str, overwrite: bool) -> tuple[int,int,int,list[str]]:
#     """
#     返回 (downloaded_now, hard_failures, soft_skips, names)
#     - 仅扫描 .obj 的“直接父目录”，不进入子目录
#     - 同名=必抓（不限扩展，排除 .obj 本体）
#     - 非同名：父目录内的图片若数量 ≤ 阈值(默认 48) 也抓
#     - 有效性校验：不合格会删除文件，并按“硬/软”分别计数
#     """
#     downloaded = 0; hard_failed = 0; soft_skipped = 0; names: list[str] = []
#     obj_stem = mesh_path.stem.lower()

#     def _record(dst: Path, existed_before: bool):
#         nonlocal downloaded, names
#         if overwrite or not existed_before:
#             downloaded += 1
#             try: names.append(str(dst.relative_to(mesh_path.parent)))
#             except Exception: names.append(dst.name)

#     def _on_bad(dst: Path, required: bool):
#         nonlocal hard_failed, soft_skipped
#         try: dst.unlink(missing_ok=True)
#         except Exception: pass
#         if required: hard_failed += 1
#         else:        soft_skipped += 1

#     parsed = _github_parse_url_inline(source_url)
#     if parsed:
#         user, repo, sha, dir_path, _ = parsed
#         parent_files = _github_list_files_in_dir_inline(user, repo, sha, dir_path)

#         same_name = [p for p in parent_files
#                      if _core_stem(p) == obj_stem and Path(p).suffix.lower() != ".obj"]
#         # 只看父目录的非同名图片
#         imgs = [p for p in parent_files if any(p.lower().endswith(ext) for ext in IMG_EXT_INLINE)]
#         if len(imgs) > IMG_BULK_THRESHOLD_INLINE:
#             imgs = []

#         if not (same_name or imgs):
#             # 列目录失败就“猜同名”
#             base = f"https://raw.githubusercontent.com/{user}/{repo}/{sha}/" + (dir_path + "/" if dir_path else "")
#             for ext in (".mtl",".json",".meta",".obj.import",".png",".jpg",".jpeg",".urdf",".ply", ".tga", ".tif", ".tiff", ".exr", ".bmp", ".dds"):
#                 rel = f"{mesh_path.stem}{ext}"
#                 dst = (mesh_path.parent / rel); dst.parent.mkdir(parents=True, exist_ok=True)
#                 existed = dst.exists()
#                 url = _join_url_dir_inline(base, rel)
#                 safe_download(url, dst, overwrite=overwrite)
#                 if _is_valid_downloaded_inline(dst):
#                     _record(dst, existed)
#                 else:
#                     # 同名：.mtl 视为 required，其它视为 soft
#                     _on_bad(dst, required=(ext==".mtl"))
#             return downloaded, hard_failed, soft_skipped, names

#         # 先同名（.mtl 计 hard，其它计 soft），再少量图片（soft）
#         for p in same_name + imgs:
#             raw = f"https://raw.githubusercontent.com/{user}/{repo}/{sha}/{quote(p)}"
#             rel = p[len(dir_path)+1:] if (dir_path and p.startswith(dir_path + "/")) else Path(p).name
#             dst = (mesh_path.parent / rel); dst.parent.mkdir(parents=True, exist_ok=True)
#             existed = dst.exists()
#             safe_download(raw, dst, overwrite=overwrite)
#             if _is_valid_downloaded_inline(dst):
#                 _record(dst, existed)
#             else:
#                 req = (Path(p).suffix.lower()==".mtl")  # 只有 .mtl 视为硬失败
#                 _on_bad(dst, required=req)
#         return downloaded, hard_failed, soft_skipped, names

#     # —— Generic：只看直接父目录（如果可列）——
#     parent = _base_url_inline(source_url)
#     parent_files = _generic_list_files_in_dir_inline(parent)

#     same_name = [n for n in parent_files
#                  if _core_stem(n) == obj_stem and Path(n).suffix.lower() != ".obj"]
#     imgs = [n for n in parent_files if any(n.lower().endswith(ext) for ext in IMG_EXT_INLINE)]
#     if len(imgs) > IMG_BULK_THRESHOLD_INLINE:
#         imgs = []

#     if not (same_name or imgs):
#         # 不可列目录时，直接猜同名
#         for ext in (".mtl",".json",".meta",".obj.import",".png",".jpg",".jpeg"):
#             rel = f"{mesh_path.stem}{ext}"
#             dst = (mesh_path.parent / rel); dst.parent.mkdir(parents=True, exist_ok=True)
#             existed = dst.exists()
#             url = _join_url_dir_inline(parent, rel)
#             safe_download(url, dst, overwrite=overwrite)
#             if _is_valid_downloaded_inline(dst):
#                 _record(dst, existed)
#             else:
#                 _on_bad(dst, required=(ext==".mtl"))
#         return downloaded, hard_failed, soft_skipped, names

#     for rel in same_name + imgs:
#         dst = (mesh_path.parent / rel); dst.parent.mkdir(parents=True, exist_ok=True)
#         existed = dst.exists()
#         url = _join_url_dir_inline(parent, rel)
#         safe_download(url, dst, overwrite=overwrite)
#         if _is_valid_downloaded_inline(dst):
#             _record(dst, existed)
#         else:
#             req = (Path(rel).suffix.lower()==".mtl")
#             _on_bad(dst, required=req)
#     return downloaded, hard_failed, soft_skipped, names

def _write_blender_helper_script():
    """Ensure Blender helper script exists next to this file. Returns its path."""
    from pathlib import Path
    helper_path = Path(__file__).with_name('fbx_collect_textures_blender.py')
    if helper_path.exists():
        return helper_path
    # 在 _write_blender_helper_script() 里，替换生成代码的相应部分：
    code = '\n'.join([
        'import bpy, sys, os, json, shutil, re',
        '',
        'def _clean_path_for_manifest(fp: str) -> str:',
        '    """把 Windows 盘符/UNC/越级路径清洗为可读相对形式（优先 basename）。"""',
        '    s = (fp or "").strip().replace("\\\\","/")',
        '    if not s:',
        '        return s',
        '    # 盘符或 UNC：仅保留文件名',
        '    if re.match(r"^[A-Za-z]:/", s) or s.startswith("//") or s.startswith("\\\\\\\\"):',
        '        return os.path.basename(s)',
        '    # Blender 在 *nix 上可能把 C:/... 当相对片段拼到前缀目录后面，这里剥离掉',
        '    if "/C:/" in s or s.startswith("C:/") or s.startswith("C:"):',
        '        return os.path.basename(s)',
        '    # 越级也只留文件名',
        '    if "/../" in f"/{s}":',
        '        return os.path.basename(s)',
        '    # 至多保留最后两段，避免把很长的上游路径写进清单',
        '    parts = [p for p in s.split("/") if p]',
        '    if len(parts) >= 2:',
        '        return "/".join(parts[-2:])',
        '    return s',
        '',
        'def main():',
        '    argv = sys.argv',
        '    if "--" not in argv or len(argv[argv.index("--")+1:]) < 2:',
        '        print("USAGE: blender --background --python fbx_collect_textures_blender.py -- <fbx_path> <out_dir>")',
        '        return',
        '    fbx_path, out_dir = argv[argv.index("--")+1:][0], argv[argv.index("--")+1:][1]',
        '    out_dir = os.path.abspath(out_dir)',
        '    os.makedirs(out_dir, exist_ok=True)',
        '',
        '    bpy.ops.wm.read_factory_settings(use_empty=True)',
        '    bpy.ops.import_scene.fbx(filepath=os.path.abspath(fbx_path))',
        '',
        '    exported = []',
        '    referenced = []',
        '',
        '    for img in list(bpy.data.images):',
        '        try:',
        '            fp = img.filepath if img.filepath else ""',
        '            try:',
        '                abspath = bpy.path.abspath(fp) if fp else ""',
        '            except Exception:',
        '                abspath = fp',
        '            # 名称一律用解析后的绝对路径 basename；没有扩展就补 .png',
        '            name = os.path.basename(abspath) if abspath else (img.name or "Image")',
        '            if not os.path.splitext(name)[1]:',
        '                name += ".png"',
        '',
        '            # 清洗后的展示路径 + 保留原始原样路径（便于排查）',
        '            clean_fp = _clean_path_for_manifest(fp)',
        '            referenced.append({"name": name, "filepath": clean_fp, "orig": fp})',
        '',
        '            dst = os.path.join(out_dir, os.path.basename(name))',
        '            if abspath and os.path.exists(abspath):',
        '                if not os.path.exists(dst):',
        '                    shutil.copy2(abspath, dst)',
        '                exported.append(os.path.basename(dst))',
        '            else:',
        '                try:',
        '                    base, ext = os.path.splitext(name)',
        '                    if not ext or ext.lower() not in [".png",".jpg",".jpeg",".tga",".bmp",".tiff",".exr",".dds",".webp"]:',
        '                        name = base + ".png"',
        '                        dst = os.path.join(out_dir, name)',
        '                        img.file_format = "PNG"',
        '                    else:',
        '                        dst = os.path.join(out_dir, name)',
        '                    img.filepath_raw = dst',
        '                    img.save()',
        '                    exported.append(os.path.basename(dst))',
        '                except Exception as ee:',
        '                    print("WARN: could not export image", img.name, "->", ee)',
        '        except Exception as e:',
        '            print("WARN: failed to process image", getattr(img, "name", "<noname>"), "->", e)',
        '',
        '    with open(os.path.join(out_dir, "textures_manifest.json"), "w", encoding="utf-8") as f:',
        '        json.dump({"exported": exported, "referenced": referenced}, f, ensure_ascii=False, indent=2)',
        '',
        'if __name__ == "__main__":',
        '    main()',
    ])

    helper_path.write_text(code, encoding='utf-8')
    return helper_path


def _blender_list_and_export_images(fbx_file: Path, out_dir: Path) -> tuple[list[str], list[str]]:
    """Use Blender (if available) to list referenced images and export embedded/existing ones.
    Returns (referenced_names, exported_names)."""
    import shutil as _shutil, subprocess, json
    blender = _shutil.which('blender') or _shutil.which('blender.exe')
    if not blender:
        return [], []
    helper = _write_blender_helper_script()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [blender, '--background', '--python', str(helper), '--', str(fbx_file), str(out_dir)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        manifest = out_dir / 'textures_manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding='utf-8'))
            ref = [ (r.get('name') or '').strip() for r in data.get('referenced',[]) if (r.get('name') or '').strip() ]
            exp = [ e for e in data.get('exported',[]) if e ]
            return ref, exp
        return [], []
    except Exception as e:
        print(f'    ⚠ Blender listing/export failed: {e}')
        return [], []
    
from urllib.parse import quote, unquote
def _github_raw_url_inline(user: str, repo: str, sha: str, rel_path: str) -> str:
     # rel_path 可能已包含 %20 等，先解码为“原字符串”，再分段编码一次
    rel_decoded = unquote(rel_path)
    rel = "/".join(quote(seg, safe="") for seg in rel_decoded.split("/") if seg)
    return f"https://raw.githubusercontent.com/{user}/{repo}/{sha}/{rel}"

# def _github_search_texture_in_repo_inline(user: str, repo: str, sha: str,
#                                          start_dir: str, filename: str,
#                                          subdirs=('Textures','textures','Materials','materials','images','Images','maps','Maps')) -> list[str]:
#     """
#     在 GitHub 仓库（固定提交/分支 sha）内，用“文件名”在若干相邻目录中查找。
#     返回可下载的 raw URL 列表（就近优先，最多一层子目录，不做深度递归）。
#     """
#     cand_relpaths = []
#     seen = set()

#     def list_dir(d: str) -> list[str]:
#         try:
#             return _github_list_files_in_dir_inline(user, repo, sha, d)
#         except Exception:
#             return []

#     # 1) 当前目录
#     for p in list_dir(start_dir):
#         if Path(p).name.lower() == filename.lower() and p not in seen:
#             cand_relpaths.append(p); seen.add(p)

#     # 2) 当前目录下常见材质子目录
#     for sd in subdirs:
#         d2 = (start_dir + "/" + sd) if start_dir else sd
#         for p in list_dir(d2):
#             if Path(p).name.lower() == filename.lower() and p not in seen:
#                 cand_relpaths.append(p); seen.add(p)

#     # 3) 父目录（再试一层子目录）
#     parent = "/".join([seg for seg in (start_dir or "").split("/") if seg][:-1])
#     if parent or parent == "":
#         for p in list_dir(parent):
#             if Path(p).name.lower() == filename.lower() and p not in seen:
#                 cand_relpaths.append(p); seen.add(p)
#         for sd in subdirs:
#             d3 = (parent + "/" + sd) if parent else sd
#             for p in list_dir(d3):
#                 if Path(p).name.lower() == filename.lower() and p not in seen:
#                     cand_relpaths.append(p); seen.add(p)

#     # 4) 仓库根（再试一层子目录）
#     root = ""
#     for p in list_dir(root):
#         if Path(p).name.lower() == filename.lower() and p not in seen:
#             cand_relpaths.append(p); seen.add(p)
#     for sd in subdirs:
#         for p in list_dir(sd):
#             if Path(p).name.lower() == filename.lower() and p not in seen:
#                 cand_relpaths.append(p); seen.add(p)
#     # 你现有的非递归就近搜索结束后，紧接着加上这段兜底：
#     # 在 _github_search_texture_in_repo_inline(...) 的返回 cand_relpaths 之前加：
#     if not cand_relpaths:
#         bases = [start_dir, "/".join([seg for seg in (start_dir or "").split("/") if seg][:-1]), ""]
#         for base in bases:
#             for rp in _github_list_files_recursive_inline(user, repo, sha, base, max_depth=GITHUB_ENUM_MAX_DEPTH):
#                 if Path(rp).name.lower() == filename.lower():
#                     cand_relpaths.append(rp)
#                     if len(cand_relpaths) >= 30:  # 给个上限，避免太多
#                         break
#             if cand_relpaths:
#                 break

#     return [_github_raw_url_inline(user, repo, sha, rp) for rp in cand_relpaths]
def _github_search_texture_in_repo_inline(user: str, repo: str, sha: str,
                                         start_dir: str, filename: str,
                                         subdirs=('Textures','textures','Materials','materials','images','Images','maps','Maps')) -> list[str]:
    """
    在 GitHub 仓库（固定提交/分支 sha）内，用“文件名”在若干相邻目录中查找。
    返回可下载的 raw URL 列表（就近优先；先用 Tree API 缓存全局过滤；再最多一层子目录；最后再递归兜底）。
    """
    cand_relpaths = []
    seen = set()

    # ====== 新增：先用 Tree API 缓存的全仓库清单做一次“按 basename 快速筛选” ======
    try:
        files_full = _github_tree_api_list_all_files_inline(user, repo, sha) or []
        low = filename.lower()
        # 先限制“起点目录附近”，避免全仓过度命中；没有 start_dir 时允许仓库根
        bases = [start_dir]
        parent = "/".join([seg for seg in (start_dir or "").split("/") if seg][:-1])
        if parent or parent == "":  # 允许根
            bases += [parent, ""]
        hits = []
        for rp in files_full:
            bn = Path(rp).name.lower()
            if bn == low and any((not b and True) or rp.startswith(b + "/") for b in bases if b is not None):
                hits.append(rp)
                if len(hits) >= 30:
                    break
        if hits:
            return [_github_raw_url_inline(user, repo, sha, rp) for rp in hits]
    except Exception:
        pass
    # ====== 新增结束 ======

    def list_dir(d: str) -> list[str]:
        try:
            return _github_list_files_in_dir_inline(user, repo, sha, d)
        except Exception:
            return []

    # 1) 当前目录
    for p in list_dir(start_dir):
        if Path(p).name.lower() == filename.lower() and p not in seen:
            cand_relpaths.append(p); seen.add(p)

    # 2) 当前目录下常见材质子目录
    for sd in subdirs:
        d2 = (start_dir + "/" + sd) if start_dir else sd
        for p in list_dir(d2):
            if Path(p).name.lower() == filename.lower() and p not in seen:
                cand_relpaths.append(p); seen.add(p)

    # 3) 父目录（再试一层子目录）
    parent = "/".join([seg for seg in (start_dir or "").split("/") if seg][:-1])
    if parent or parent == "":
        for p in list_dir(parent):
            if Path(p).name.lower() == filename.lower() and p not in seen:
                cand_relpaths.append(p); seen.add(p)
        for sd in subdirs:
            d3 = (parent + "/" + sd) if parent else sd
            for p in list_dir(d3):
                if Path(p).name.lower() == filename.lower() and p not in seen:
                    cand_relpaths.append(p); seen.add(p)

    # 4) 仓库根（再试一层子目录）
    root = ""
    for p in list_dir(root):
        if Path(p).name.lower() == filename.lower() and p not in seen:
            cand_relpaths.append(p); seen.add(p)
    for sd in subdirs:
        for p in list_dir(sd):
            if Path(p).name.lower() == filename.lower() and p not in seen:
                cand_relpaths.append(p); seen.add(p)

    # 5) 兜底：递归枚举（受 GITHUB_ENUM_MAX_DEPTH 限制），覆盖非常规目录如 “Materials and Textures”
    if not cand_relpaths:
        bases = [start_dir, "/".join([seg for seg in (start_dir or "").split("/") if seg][:-1]), ""]
        for base in bases:
            for rp in _github_list_files_recursive_inline(user, repo, sha, base, max_depth=GITHUB_ENUM_MAX_DEPTH):
                if Path(rp).name.lower() == filename.lower():
                    cand_relpaths.append(rp)
                    if len(cand_relpaths) >= 30:
                        break
            if cand_relpaths:
                break

    return [_github_raw_url_inline(user, repo, sha, rp) for rp in cand_relpaths]

# === End helpers ===

def safe_extract(archive_path: Path, out_dir: Path, overwrite: bool=False):
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        return
    ensure_dir(out_dir)

    low = archive_path.name.lower()
    if low.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(out_dir)
        return
    if any(low.endswith(ext) for ext in [".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz"]):
        mode = "r"
        if low.endswith((".tar.gz", ".tgz")): mode = "r:gz"
        elif low.endswith((".tar.xz", ".txz")): mode = "r:xz"
        elif low.endswith((".tar.bz2", ".tbz")): mode = "r:bz2"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(out_dir)
        return

    # 其它压缩格式（7z/rar等），尝试调用系统 7z
    try:
        subprocess.run(["7z", "x", str(archive_path), f"-o{str(out_dir)}", "-y"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        raise RuntimeError(f"不支持的压缩格式，且无法用 7z 解压：{archive_path}")

def write_or_update_metadata(metadata_csv: Path, rows: List[Dict[str, str]]):
    """
    将 rows 写入 metadata.csv。若已存在，则按 sha256 去重/更新。
    """
    existing: Dict[str, Dict[str, str]] = {}
    if metadata_csv.exists():
        with metadata_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "sha256" in r and r["sha256"]:
                    existing[r["sha256"]] = r

    for r in rows:
        existing[r["sha256"]] = r

    fieldnames = ["sha256", "local_path", "category", "shape_key", "source_url", "rendered"]
    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=",",          # avoid all in one cell
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n"
        )
        writer.writeheader()
        for r in existing.values():
            # 统一 string 化
            out = {k: ("" if r.get(k) is None else str(r.get(k))) for k in fieldnames}
            writer.writerow(out)

def _open_csv_writer(path: Path, fieldnames, delimiter=","):
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter,
                            quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writeheader()
    return f, writer

def main():
    args = parse_args()
    global DL_WORKERS
    DL_WORKERS = int(getattr(args, "dl_workers", 16))
    ids = load_id_set(args.ids_json, args.ids_txt)
    png_root = Path(args.png_root).expanduser().resolve()
    download_root = Path(args.download_root).expanduser().resolve() if args.download_root else png_root
  
    # 推断类别
    if args.categories:
        categories = str(args.categories[0])
    else:
        categories = guess_categories_from_png_root(png_root)
    if not categories:
        print("未在 png_root 下找到任何类别文件夹。")
        sys.exit(1)
    metadata_csv = Path(args.metadata_csv).expanduser().resolve() if args.metadata_csv else (download_root / categories /"metadata.csv")
    report_csv = Path(args.report_csv).expanduser().resolve() if args.report_csv else (download_root / categories /"download_report.csv")
    if not ids:
        print("未从 --ids_json / --ids_txt 读取到任何 shape_id（形如 'subA/subB'）。")
        sys.exit(1)

    # 加载映射
    cat2map = load_maps(args.maps_json, categories=None if len(categories) > 1 else categories)
    # 解析类别集合
    if "__SINGLE__" in cat2map:
        # 单类映射，必须由 --categories 明确类名（通常就是 mug/teapot 等）
        if not args.categories or len(args.categories) != 1:
            raise ValueError("maps_json 是单类结构，请用 --categories 明确该类名（唯一）。")
        cats = args.categories
    else:
        # 多类映射；若 --categories 提供则只在这些类中查，否则全部类中查
        cats = args.categories if args.categories else sorted(cat2map.keys())
    # 收集实例
    # instances = list(iter_instances_from_png(png_root, categories))
    # if args.debug_limit:
    #     instances = instances[:args.debug_limit]

    rows = []
    report_rows = []  # 每个实例一行，包含成功/失败与原因
    summary = {
        "entries": 0,
        "tex_refs": 0,
        "tex_downloaded": 0,
        "tex_missing": 0,
        "tex_repo_missing": 0,
        "tex_repo_failed": 0,
    }
    total = len(ids)
    print(f"将按 id 列表处理 {total} 个实例（仅索引 final_rotation 中对应链接）...")
    missing = 0
    for idx, shape_key in enumerate(ids, 1):
        # 在选定的类别集合里查 url
        url = None; cat_hit = None
        for cat in cats:
            m = cat2map[cat if cat in cat2map else "__SINGLE__"]
            if shape_key in m:
                url = m[shape_key]; cat_hit = (cat if cat in cat2map else cats[0])
                break
        if not url:
            print(f"[{idx}/{total}] ✘ 未在 maps_json 中找到链接：{shape_key}")
            missing += 1
            continue

        if not url:
            print(f"[{idx}/{total}] ✘ 映射缺失：{cat} -> {shape_key}")
            continue

        # 目标实例根：download_root/<cat>/<subA>/<subB>
        subA, subB = shape_key.split("/", 1)
        inst_root = download_root / cat / subA / subB
        ensure_dir(inst_root)

        # 下载到 raw/
        fname = Path(urlparse(url).path).name or "asset"
        raw_dir = inst_root / "raw"
        ensure_dir(raw_dir)
        # tree cache 
        cache_dir = (raw_dir / ".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _GITHUB_TREE_CACHE_FILE = cache_dir / "github_tree_cache.json"
        _git_tree_cache_load(_GITHUB_TREE_CACHE_FILE)
        atexit.register(_git_tree_cache_save, _GITHUB_TREE_CACHE_FILE)

        dst_file = raw_dir / fname
        # --- 新增: 跳过检查 ---
        if not args.overwrite:
            existing = []
            for ext in ("*.glb", "*.fbx", "*.obj"):
                existing.extend(raw_dir.glob(ext))
            if existing:
                print(f"[{idx}/{total}] SKIP 已存在模型: {cat}/{shape_key} -> {existing[0].name}")
                base_row = dict(
                    category=cat,
                    shape_key=shape_key,
                    url=url,
                    status="OK",
                    reason="skip_existing",
                    saved_file="",
                    extracted_mesh="",
                                    **{
                        "tex_kind": "",
                        "tex_refs": 0,
                        "tex_downloaded": 0,
                        "tex_invalid": 0,
                        "tex_missing": 0,
                        "tex_exported_embedded": 0,
                        "tex_filenames": "",
                        "tex_invalid_names": "",
                        "tex_sources": "",
                        "tex_index_check": "",
                        "tex_index_blender": 0,
                        "tex_index_regex": 0,
                        "tex_missing_names": "",
                        "tex_missing_reason": "",
                        "tex_repo_missing": 0,
                        "tex_repo_missing_names": "",
                        "tex_repo_failed": 0,
                        "tex_repo_failed_names": "",
                    },

                )
                report_rows.append(base_row)
                continue
        base_row = dict(
                    category=cat,
                    shape_key=shape_key,
                    url=url,
                    status="Fail",
                    reason="",
                    saved_file="",
                    extracted_mesh="",
                      **{
                        "tex_kind": "",
                        "tex_refs": 0,
                        "tex_downloaded": 0,
                        "tex_invalid": 0,
                        "tex_missing": 0,
                        "tex_exported_embedded": 0,
                        "tex_filenames": "",
                        "tex_invalid_names": "",
                        "tex_sources": "",
                        "tex_index_check": "",
                        "tex_index_blender": 0,
                        "tex_index_regex": 0,
                        "tex_missing_names": "",
                        "tex_missing_reason": "",
                        "tex_repo_missing": 0,
                        "tex_repo_missing_names": "",
                        "tex_repo_failed": 0,
                        "tex_repo_failed_names": "",
                    },
            )
        try:
            original_url = url.strip()
            resolved_url = fix_github_blob(original_url)
            dst_file = safe_download(resolved_url, dst_file, overwrite=args.overwrite)
        except Exception as e:
            print(f"[{idx}/{total}] ✘ 下载失败：{url} ；错误：{e}")
            base_row["reason"] = f"download_error:{e}"
            report_rows.append(base_row)
            continue

        # 计算 sha256（用下载原文件）
        try:
            sha = sha256_of_file(dst_file)
        except Exception as e:
            print(f"[{idx}/{total}] ✘ sha256 失败：{dst_file} ；错误：{e}")
            continue

        mesh_path: Optional[Path] = None
        
        tex_report = None
        if not is_archive(dst_file.name):
            tex_report = ensure_complete_asset_anyformat(dst_file, resolved_url, raw_dir, overwrite=args.overwrite)

        if not args.skip_extract and is_archive(dst_file.name):
            # 解压到 mesh/
            mesh_dir = inst_root / "mesh"
            try:
                safe_extract(dst_file, mesh_dir, overwrite=args.overwrite)
            except Exception as e:
                print(f"[{idx}/{total}] ✘ 解压失败：{dst_file} ；错误：{e}")
                base_row["saved_file"] = str(dst_file)
                base_row["reason"] = f"extract_error:{e}"
                report_rows.append(base_row)
                continue
            mesh_path = pick_mesh_file(mesh_dir)
            if mesh_path and tex_report is None:
                tex_report = ensure_complete_asset_anyformat(mesh_path, resolved_url, raw_dir, overwrite=args.overwrite)
        else:
            # 非压缩或选择跳过解压：直接把下载文件当作候选
            if any(dst_file.name.lower().endswith(ext) for ext in MESH_EXT_PRIORITY):
                mesh_path = dst_file
            else:
                # 万一下载的是单个目录（极少数情况），也尝试找
                mesh_path = pick_mesh_file(inst_root)

        if not mesh_path:
            print(f"[{idx}/{total}] ✘ 未找到可渲染网格：{inst_root}（url={url}）")
            base_row["saved_file"] = str(dst_file)
            base_row["reason"] = "no_mesh_found"
            report_rows.append(base_row)
            continue

        row = {
            "sha256": sha,
            "local_path": str(mesh_path),
            "category": cat,
            "shape_key": shape_key,
            "source_url": url,
            "rendered": "False",
        }
        rows.append(row)
        print(f"[{idx}/{total}] ✓ {cat}/{shape_key}  ->  {mesh_path.name}")

        # 成功时：
        base_row.update({
            "status": "ok",
            "reason": "ok",
            "saved_file": str(dst_file),
            "extracted_mesh": str(mesh_path),
        })
        if 'tex_report' in locals() and tex_report is not None:
            missing_names = tex_report.get("missing_names", []) or []
            repo_missing_names = tex_report.get("repo_missing_names", []) or []
            repo_failed = tex_report.get("found_but_failed", []) or []
            missing_reason_map = tex_report.get("missing_reason_map", {}) or {}

            base_row.update({
                "tex_kind": tex_report.get("kind",""),
                "tex_refs": tex_report.get("refs_total", 0),
                "tex_downloaded": tex_report.get("downloaded", 0),
                "tex_invalid": tex_report.get("invalid_downloaded", 0),
                "tex_missing": len(missing_names),
                "tex_exported_embedded": tex_report.get("exported_embedded", 0),
                "tex_filenames": ";".join(tex_report.get("fetched_names", [])),
                "tex_invalid_names": ";".join(tex_report.get("fetched_invalid_names", [])),
                "tex_sources": ";".join(tex_report.get("fetched_sources", [])),
                "tex_index_check": tex_report.get("tex_index_check",""),
                "tex_index_blender": tex_report.get("tex_index_blender_count",0),
                "tex_index_regex": tex_report.get("tex_index_regex_count",0),
                "tex_missing_names": ";".join(missing_names),
                "tex_missing_reason": ";".join(f"{k}:{v}" for k, v in missing_reason_map.items()),
                "tex_repo_missing": len(repo_missing_names),
                "tex_repo_missing_names": ";".join(repo_missing_names),
                "tex_repo_failed": len(repo_failed),
                "tex_repo_failed_names": ";".join(repo_failed),
            })

            summary["entries"] += 1
            summary["tex_refs"] += int(tex_report.get("refs_total", 0) or 0)
            summary["tex_downloaded"] += int(tex_report.get("downloaded", 0) or 0)
            summary["tex_missing"] += len(missing_names)
            summary["tex_repo_missing"] += len(repo_missing_names)
            summary["tex_repo_failed"] += len(repo_failed)

        report_rows.append(base_row)

    if rows:
        write_or_update_metadata(metadata_csv, rows)
        print(f"已写出/更新 metadata：{metadata_csv} （新增 {len(rows)} 行）")
    else:
        print("没有新增可用实例写入 metadata。")
    if report_rows:
        # 你可以把分隔符设为 "," 或 ";" 或 "\t"
        rep_fields = [
            "category","shape_key","url","status","reason","saved_file","extracted_mesh",
            "tex_kind","tex_refs","tex_downloaded","tex_invalid","tex_missing",
            "tex_exported_embedded","tex_filenames","tex_invalid_names","tex_sources",
            "tex_index_check","tex_index_blender","tex_index_regex",
            "tex_missing_names","tex_missing_reason","tex_repo_missing","tex_repo_missing_names",
            "tex_repo_failed","tex_repo_failed_names"
        ]

        f_rep, wr = _open_csv_writer(report_csv, rep_fields, delimiter=",")  # 逗号 CSV
        for r in report_rows: wr.writerow(r)
        f_rep.close()
        print(f"已写出下载报告：{report_csv}（共 {len(report_rows)} 条，含失败原因）")

    if summary["entries"]:
        print("下载贴图摘要：")
        print(f"  · 处理模型: {summary['entries']} 条")
        print(f"  · 索引贴图总数: {summary['tex_refs']}，成功补齐: {summary['tex_downloaded']}，缺失: {summary['tex_missing']}")
        print(f"  · 仓库缺失: {summary['tex_repo_missing']}，下载失败: {summary['tex_repo_failed']}")
if __name__ == "__main__":
    main()
