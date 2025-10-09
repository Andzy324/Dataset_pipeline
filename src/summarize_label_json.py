#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize per-label rotation JSONs produced by your split step.

Directory layout assumed:
  <rot_dir>/<label>/final_shape_rotations.json

Each JSON can be either:
  - New schema: { "<sha>": { "rotation_matrix": [...], "label": [...], ... }, ... }
  - Old schema: { "shape_rotations": { "<sha>": {"rotation_matrix": [...]}, ... }, ... }

Outputs:
  1) Console table with: label, count, invalid_sha, missing_R
  2) Optional CSV (--output summary.csv)

Extras:
  - Detect duplicated sha across labels (sha appears in >1 labels)
"""

import argparse, json, sys, re
from pathlib import Path
from typing import Dict, Any, Tuple, List

SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.I)

def _labels_of(info: Dict[str, Any]):
    labs = info.get("label", []) or []
    if isinstance(labs, str):
        labs = [labs]
    return [str(x) for x in labs if x]

def _label_matches_mode(labels: list[str], folder_label: str, mode: str) -> bool:
    if mode == "single":
        return len(labels) == 1 and labels[0] == folder_label
    if mode == "primary":
        return len(labels) >= 1 and labels[0] == folder_label
    # 'any'
    return folder_label in labels

def load_map(json_path: Path) -> Dict[str, Dict[str, Any]]:
    """Return a mapping {sha: info} from either new or old schema."""
    try:
        J = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] cannot read {json_path}: {e}", file=sys.stderr)
        return {}
    if isinstance(J, dict) and isinstance(J.get("shape_rotations"), dict):
        M = J["shape_rotations"]
        return {k: v for k, v in M.items() if isinstance(v, dict)}
    # new schema: top-level <sha> → info (filter out non-dict meta like export_info)
    return {k: v for k, v in (J.items() if isinstance(J, dict) else []) if isinstance(v, dict)}

def is_sha(s: str) -> bool:
    return bool(SHA256_RE.match(s or ""))

def has_valid_R(info: Dict[str, Any]) -> bool:
    R = info.get("rotation_matrix", info.get("R"))
    if R is None:
        return False
    # quick shape check: 3x3 list-like
    try:
        return (len(R) == 3 and all(len(row) == 3 for row in R))
    except Exception:
        return False

def summarize(rot_dir: Path, filename: str = "final_shape_rotations.json") -> Tuple[List[Tuple[str,int,int,int]], Dict[str,List[str]]]:
    rows = []
    sha2labels: Dict[str, List[str]] = {}
    if not rot_dir.exists():
        raise FileNotFoundError(f"--rot_dir not found: {rot_dir}")
    for lab_dir in sorted([d for d in rot_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]):
        label = lab_dir.name
        fp = lab_dir / filename
        if not fp.exists():
            # 容错：如果没有指定文件名，尝试找单个 .json
            cand = list(lab_dir.glob("*.json"))
            if len(cand) == 1:
                fp = cand[0]
            else:
                print(f"[WARN] no JSON under {lab_dir} (expect {filename})", file=sys.stderr)
                rows.append((label, 0, 0, 0))
                continue

        M = load_map(fp)
        if not M:
            rows.append((label, 0, 0, 0))
            continue

        total = len(M)
        bad_sha = sum(0 if is_sha(k) else 1 for k in M.keys())
        missing_R = sum(0 if has_valid_R(v) else 1 for v in M.values())

        # collect duplicates across labels
        for k in M.keys():
            if not is_sha(k):  # 只统计合法 sha 的跨类重复
                continue
            sha2labels.setdefault(k, []).append(label)

        rows.append((label, total, bad_sha, missing_R))
    return rows, sha2labels

# --- update print_table to show the new column when有意义 ---
def print_table(rows):
    # rows: (label, count, invalid_sha, missing_R, mismatches)
    rows_sorted = sorted(rows, key=lambda r: (-r[1], r[0].lower())) if rows else []
    w_label = max(5, *(len(r[0]) for r in rows_sorted)) if rows_sorted else 5
    show_mismatch = any(r[4] != 0 for r in rows_sorted) or any(len(r) >= 5 for r in rows_sorted)
    header = f"{'label'.ljust(w_label)} | {'count':>6} | {'invalid_sha':>11} | {'missing_R':>10}"
    if show_mismatch:
        header += " | " + f"{'label_mismatch':>14}"
    print(header)
    print("-" * len(header))
    total = 0
    for r in rows_sorted:
        lab, cnt, bad, miss = r[0], r[1], r[2], r[3]
        total += cnt
        line = f"{lab.ljust(w_label)} | {cnt:6d} | {bad:11d} | {miss:10d}"
        if show_mismatch:
            mm = r[4] if len(r) >= 5 else 0
            line += " | " + f"{mm:14d}"
        print(line)
    print("-" * len(header))
    footer = f"{'TOTAL'.ljust(w_label)} | {total:6d} | {'':>11} | {'':>10}"
    if show_mismatch:
        footer += " | " + f"{'':>14}"
    print(footer)

def write_csv(rows: List[Tuple[str,int,int,int]], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("label,count,invalid_sha,missing_R\n")
        for lab, cnt, bad, miss in rows:
            f.write(f"{lab},{cnt},{bad},{miss}\n")
    print(f"[OK] wrote CSV → {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot_dir", required=True, type=Path, help="Root of per-label JSONs (each at <rot_dir>/<label>/final_shape_rotations.json).")
    ap.add_argument("--filename", default="final_shape_rotations.json", help="JSON filename in each label dir (default: final_shape_rotations.json).")
    ap.add_argument("--output", type=Path, default=None, help="Optional CSV path to write summary.")
    ap.add_argument("--show_dups", action="store_true", help="Show duplicated sha that appear in multiple labels.")
    ap.add_argument("--expect_mode", choices=["any","primary","single"], default="any",
                help="Validate that each entry's label matches the folder label under the given rule.")
    args = ap.parse_args()

    rows, sha2labels = summarize(args.rot_dir, args.filename, args.expect_mode)
    print_table(rows)

    if args.output:
        write_csv(rows, args.output)

    if args.show_dups:
        dups = {k:v for k,v in sha2labels.items() if len(set(v)) > 1}
        if dups:
            print(f"\n[DUPE] sha appearing in multiple labels: {len(dups)}")
            # 打印前若干条示例
            shown = 0
            for sha, labs in dups.items():
                print(f"  {sha}: {sorted(set(labs))}")
                shown += 1
                if shown >= 20:
                    print("  ...")
                    break
        else:
            print("\n[DUPE] no cross-label duplicates found.")

if __name__ == "__main__":
    main()
