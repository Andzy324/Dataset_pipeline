#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split a global label_json (new or old schema) into per-label files:
  <OUT_DIR>/<label>/final_shape_rotations.json

- "Collect-only": keep each entry's fields as-is (e.g., rotation_matrix, label, etc.)
- No wrapping under "shape_rotations"; top-level is { "<sha>": { ... }, ... }
- Works even if the source uses the old schema with "shape_rotations" at top-level

Usage:
  python split_label_json_collect.py \
    --label_json /path/to/all_sha_rotation.json \
    --out_dir    /path/to/manifests/rotations \
    --labels     drill hammer mug pot   # optional; if omitted, auto-discover all labels
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable
import copy 

def _load_rotations_dict(src: Path) -> Dict[str, Dict[str, Any]]:
    """Return a dict of {sha: info} from either schema."""
    J = json.loads(Path(src).read_text(encoding="utf-8"))
    if isinstance(J, dict) and isinstance(J.get("shape_rotations"), dict):
        # old schema -> unwrap
        return {k: v for k, v in J["shape_rotations"].items() if isinstance(v, dict)}
    # new schema -> top-level sha entries; drop non-dict meta like "export_info"
    return {k: v for k, v in (J.items() if isinstance(J, dict) else []) if isinstance(v, dict)}

def _labels_of(info: Dict[str, Any]) -> Iterable[str]:
    labs = info.get("label", []) or []
    if isinstance(labs, str):
        labs = [labs]
    return [str(x) for x in labs if x]

# --- replace split_collect with the normalized version ---
def split_collect(label_json: Path, out_dir: Path, labels_whitelist=None, min_items=1,
                  label_field_mode: str = "single", audit_multilabel: bool = False) -> Dict[str, Path]:
    """
    label_field_mode:
      - 'single'  : force label to exactly [<lab>] for each per-label file
      - 'primary' : keep all labels but ensure <lab> is first
      - 'keep'    : keep original labels unchanged (legacy behavior)
    audit_multilabel:
      - If True and an entry has >1 labels originally, store them in '__labels_all'
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    R = _load_rotations_dict(label_json)

    # discover labels if none provided
    if not labels_whitelist:
        seen = set()
        for _, info in R.items():
            for lab in _labels_of(info):
                seen.add(lab)
        labels_whitelist = sorted(seen)

    out_map = {}
    for lab in labels_whitelist:
        bucket = {}
        for sha, info in R.items():
            labs = list(_labels_of(info))
            if lab not in labs:
                continue
            new_info = copy.deepcopy(info)

            if label_field_mode == "single":
                new_info["label"] = [lab]
            elif label_field_mode == "primary":
                # put the target label first; keep the rest (dedup, stable-ish)
                rest = [x for x in labs if x != lab]
                new_info["label"] = [lab] + rest
            else:  # 'keep'
                pass

            if audit_multilabel and len(labs) > 1:
                new_info["__labels_all"] = labs

            bucket[sha] = new_info

        if len(bucket) < min_items:
            continue
        dst_dir = out_dir / lab
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "final_shape_rotations.json"
        # stable key order for reproducibility
        dst.write_text(json.dumps(bucket, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        out_map[lab] = dst
        
    return out_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_json", required=True, type=Path, help="Global label_json (new or old schema).")
    ap.add_argument("--out_dir",    required=True, type=Path, help="Output root directory.")
    ap.add_argument("--labels",     nargs="*", default=None, help="Optional label whitelist (space-separated).")
    ap.add_argument("--min_items",  type=int, default=1, help="Skip writing labels with fewer than N items.")
    ap.add_argument("--label_field_mode", choices=["single", "primary", "keep"], default="single",
                help="How to write 'label' into each per-label file.")
    ap.add_argument("--audit_multilabel", action="store_true",
                help="If set, save original multiple labels to '__labels_all'.")
    
    args = ap.parse_args()

    out_map = split_collect(args.label_json, args.out_dir, args.labels, args.min_items)
    print(f"[split] wrote {len(out_map)} label files under: {args.out_dir}")
    for lab, fp in sorted(out_map.items()):
        print(f"  - {lab}: {fp}")

if __name__ == "__main__":
    main()
