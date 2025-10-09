#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_label_json.py (enhanced)
----------------------------------
Bidirectional validator & summarizer between the *base* (global) label JSON
and the *split* per-label folders.

Adds:
- Forward check  (base -> split): every (sha, label) in base exists in split
- Backward check (split -> base): every (sha, label) in split exists in base
- Intra-file `label` field validation vs. --expect_mode {any, primary, single}
- Stronger multi-label validation & audit (exact/extra/missing/mismatch)
- Optional CSV summary output

Usage:
  python summarize_label_json.py \
    --label_json /path/to/all_sha256_to_rotation.json \
    --rot_dir    /path/to/labeled_rotations \
    --expect_mode single \
    --show_dups \
    --output summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

# ------------------------
# Helpers
# ------------------------

def is_sha(s: str) -> bool:
    """Minimal sha256-like check: 64 hex chars."""
    if not isinstance(s, str) or len(s) != 64:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False

def _load_rotations_dict(src: Path) -> Dict[str, Dict[str, Any]]:
    """
    Return a dict of {sha: info} from either schema:
    - old: {"shape_rotations": {...}}   -> unwrap
    - new: {sha: info, "export_info": ...} -> filter dict values by sha keys
    """
    J = json.loads(Path(src).read_text(encoding="utf-8"))
    if isinstance(J, dict) and isinstance(J.get("shape_rotations"), dict):
        return {k: v for k, v in J["shape_rotations"].items() if isinstance(v, dict)}
    if isinstance(J, dict):
        return {k: v for k, v in J.items() if isinstance(v, dict)}
    return {}

def _labels_of(info: Dict[str, Any]) -> List[str]:
    labs = info.get("label", []) or []
    if isinstance(labs, str):
        labs = [labs]
    return [str(x) for x in labs if x]

def _label_matches_mode(labels: List[str], folder_label: str, mode: str) -> bool:
    if mode == "single":
        return len(labels) == 1 and labels[0] == folder_label
    if mode == "primary":
        return len(labels) >= 1 and labels[0] == folder_label
    # any
    return folder_label in labels

def _has_valid_R(v: Dict[str, Any]) -> bool:
    """
    Very light check: accept R in different common keys and shapes.
    """
    R = v.get("rotation_matrix") or v.get("rotation") or v.get("rot") or None
    if R is None:
        return False
    if isinstance(R, list):
        # Flatten length 9 or nested [3][3]
        if len(R) == 9:
            return True
        if len(R) == 3 and all(isinstance(r, list) and len(r) == 3 for r in R):
            return True
    return False

# ------------------------
# Core scanning
# ------------------------

def scan_split(rot_dir: Path, filename: str, expect_mode: str):
    """
    Walk per-label folders and collect stats.

    Returns:
      per_label_stats: dict[label] -> dict with {total, file, exists}
      split_membership: set[(sha, label)] observed in split
      label_mismatches: list[(label_folder, sha, labels_in_file)]
      bad_sha: dict[label] -> count of invalid sha keys
      missing_R: dict[label] -> count of entries lacking valid R
      split_labels_by_sha: dict[sha] -> set(labels) by folder presence
      audit_alllabels_by_sha: dict[sha] -> set(...) consolidated from '__labels_all'
      audit_alllabels_conflicts: list[(sha, folder_label, seen_set, new_set)]
    """
    per_label_stats = {}
    split_membership: Set[Tuple[str, str]] = set()
    label_mismatches = []
    bad_sha = {}
    missing_R = {}
    split_labels_by_sha: Dict[str, Set[str]] = {}
    audit_alllabels_by_sha: Dict[str, Set[str]] = {}
    audit_alllabels_conflicts = []

    if not rot_dir.exists():
        raise FileNotFoundError(f"--rot_dir not found: {rot_dir}")

    label_dirs = [d for d in rot_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    for lab_dir in sorted(label_dirs, key=lambda p: p.name.lower()):
        label = lab_dir.name
        fp = lab_dir / filename
        if not fp.exists():
            cand = list(lab_dir.glob("*.json"))
            if len(cand) == 1:
                fp = cand[0]
            else:
                per_label_stats[label] = dict(total=0, file=str(fp), exists=False)
                bad_sha[label] = 0
                missing_R[label] = 0
                continue

        try:
            M = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ERROR] failed to read {fp}: {e}", file=sys.stderr)
            M = {}

        total = len(M) if isinstance(M, dict) else 0
        bad_sha_cnt = 0
        missR_cnt = 0

        for sha, v in (M.items() if isinstance(M, dict) else []):
            if not is_sha(sha):
                bad_sha_cnt += 1
                continue

            labs = _labels_of(v)
            if not _label_matches_mode(labs, label, expect_mode):
                label_mismatches.append((label, sha, labs))
            if not _has_valid_R(v):
                missR_cnt += 1

            # record split membership and per-sha labels (by folder presence)
            split_membership.add((sha, label))
            split_labels_by_sha.setdefault(sha, set()).add(label)

            # consolidate audit '__labels_all' if present
            audit = v.get("__labels_all", None)
            if audit is not None:
                if isinstance(audit, str):
                    audit_set = {audit}
                elif isinstance(audit, list):
                    audit_set = {str(x) for x in audit if x}
                else:
                    audit_set = set()
                if sha not in audit_alllabels_by_sha:
                    audit_alllabels_by_sha[sha] = audit_set
                else:
                    if audit_alllabels_by_sha[sha] != audit_set:
                        audit_alllabels_conflicts.append(
                            (sha, label, set(audit_alllabels_by_sha[sha]), set(audit_set))
                        )

        per_label_stats[label] = dict(total=total, file=str(fp), exists=True)
        bad_sha[label] = bad_sha_cnt
        missing_R[label] = missR_cnt

    return (per_label_stats, split_membership, label_mismatches,
            bad_sha, missing_R, split_labels_by_sha,
            audit_alllabels_by_sha, audit_alllabels_conflicts)

def build_base_membership(label_json: Path):
    """
    Returns:
      base_labels: dict[sha] -> set(labels)
      base_membership: set[(sha, label)]
      per_label_base_counts: dict[label] -> count in base
    """
    R = _load_rotations_dict(label_json)
    base_labels: Dict[str, Set[str]] = {}
    for sha, info in R.items():
        labs = set(_labels_of(info))
        if labs:
            base_labels[sha] = labs

    base_membership: Set[Tuple[str, str]] = set()
    for sha, labs in base_labels.items():
        for lab in labs:
            base_membership.add((sha, lab))

    per_label_base_counts: Dict[str, int] = {}
    for _, lab in base_membership:
        per_label_base_counts[lab] = per_label_base_counts.get(lab, 0) + 1

    return base_labels, base_membership, per_label_base_counts

# ------------------------
# Multi-label analysis
# ------------------------

def multilabel_recon(base_labels: Dict[str, Set[str]], split_labels_by_sha: Dict[str, Set[str]]):
    """
    Compare base vs split per-sha label sets, focusing on multi-label SHAs in base.
    Returns dict with counts and example lists.
    """
    exact = []
    missing = []   # split subset of base (but not equal / possibly empty)
    extra = []     # split superset (has labels not in base)
    mismatch = []  # other inconsistencies

    base_multi = {sha for sha, labs in base_labels.items() if len(labs) > 1}
    for sha in sorted(base_multi):
        b = base_labels.get(sha, set())
        s = split_labels_by_sha.get(sha, set())
        if not s:
            missing.append((sha, sorted(list(b)), []))
            continue
        if s == b:
            exact.append((sha, sorted(list(b))))
        elif s.issubset(b):
            missing.append((sha, sorted(list(b)), sorted(list(s))))
        elif s.issuperset(b):
            extra.append((sha, sorted(list(b)), sorted(list(s))))
        else:
            mismatch.append((sha, sorted(list(b)), sorted(list(s))))

    return dict(
        counts=dict(
            base_multilabel=len(base_multi),
            exact=len(exact),
            missing=len(missing),
            extra=len(extra),
            mismatch=len(mismatch),
        ),
        examples=dict(
            exact=exact[:20],
            missing=missing[:20],
            extra=extra[:20],
            mismatch=mismatch[:20],
        )
    )

def audit_alllabels_check(base_labels: Dict[str, Set[str]], audit_alllabels_by_sha: Dict[str, Set[str]]):
    """
    Validate that '__labels_all' (if present) matches base labels for that SHA.
    Returns mismatches and missing audit cases for multi-label SHAs.
    """
    mismatches = []
    missing_audit = []
    for sha, b in base_labels.items():
        if sha in audit_alllabels_by_sha:
            if audit_alllabels_by_sha[sha] != b:
                mismatches.append((sha, sorted(list(b)), sorted(list(audit_alllabels_by_sha[sha]))))
        else:
            if len(b) > 1:
                missing_audit.append((sha, sorted(list(b))))
    return mismatches, missing_audit

# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_json", required=True, type=Path, help="Global label_json (new or old schema).")
    ap.add_argument("--rot_dir",    required=True, type=Path, help="Root of per-label folders.")
    ap.add_argument("--filename",   default="final_shape_rotations.json", help="JSON file inside each label folder.")
    ap.add_argument("--expect_mode", choices=["any","primary","single"], default="any",
                    help="Validate each entry's 'label' field against folder name.")
    ap.add_argument("--show_dups", action="store_true", help="List SHAs that appear in multiple split labels.")
    ap.add_argument("--output", type=Path, default=None, help="Optional CSV to write summary table.")
    args = ap.parse_args()

    # 1) Base
    base_labels, base_membership, per_label_base_counts = build_base_membership(args.label_json)

    # 2) Split
    (per_label_stats, split_membership, label_mismatches,
     bad_sha, missing_R, split_labels_by_sha,
     audit_alllabels_by_sha, audit_alllabels_conflicts) = scan_split(args.rot_dir, args.filename, args.expect_mode)

    # 3) Bidirectional checks
    missing_in_split = base_membership - split_membership       # base -> split
    extra_in_split   = split_membership - base_membership       # split -> base

    # 4) Per-label reconciliation table
    labels_all = sorted(set(list(per_label_base_counts.keys()) + list(per_label_stats.keys())))
    rows = []
    for lab in labels_all:
        base_cnt  = per_label_base_counts.get(lab, 0)
        split_cnt = per_label_stats.get(lab, {}).get("total", 0)
        fwd_miss  = sum(1 for (sha, L) in missing_in_split if L == lab)
        bwd_extra = sum(1 for (sha, L) in extra_in_split   if L == lab)
        mism_cnt  = sum(1 for (L, _, _) in label_mismatches if L == lab)
        bad_cnt   = bad_sha.get(lab, 0)
        missR_cnt = missing_R.get(lab, 0)
        coverage  = 0.0 if base_cnt == 0 else 100.0 * (base_cnt - fwd_miss) / base_cnt
        rows.append((lab, base_cnt, split_cnt, fwd_miss, bwd_extra, mism_cnt, bad_cnt, missR_cnt, coverage))

    # 5) Print table
    if rows:
        w_label = max(5, max(len(r[0]) for r in rows))
    else:
        w_label = 5
    header = (
        f"{'label'.ljust(w_label)} | {'base_cnt':>8} | {'split_cnt':>9} | "
        f"{'miss->split':>11} | {'extra<-split':>12} | {'label_mismatch':>14} | "
        f"{'invalid_sha':>11} | {'missing_R':>10} | {'coverage%':>9}"
    )
    print(header)
    print("-" * len(header))
    total_base = total_split = total_fwd_miss = total_bwd_extra = 0
    total_mismatch = total_bad = total_missR = 0
    for r in sorted(rows, key=lambda x: (-x[1], x[0])):
        lab, bc, sc, fm, be, mm, bd, mr, cov = r
        total_base += bc
        total_split += sc
        total_fwd_miss += fm
        total_bwd_extra += be
        total_mismatch += mm
        total_bad += bd
        total_missR += mr
        print(f"{lab.ljust(w_label)} | {bc:8d} | {sc:9d} | {fm:11d} | {be:12d} | {mm:14d} | {bd:11d} | {mr:10d} | {cov:9.2f}")
    print("-" * len(header))
    print(f"{'TOTAL'.ljust(w_label)} | {total_base:8d} | {total_split:9d} | {total_fwd_miss:11d} | {total_bwd_extra:12d} | {total_mismatch:14d} | {total_bad:11d} | {total_missR:10d} | {'':>9}")

    # 6) Duplicates in split (same SHA across multiple label folders)
    if args.show_dups:
        sha2labs = {}
        for sha, lab in split_membership:
            sha2labs.setdefault(sha, set()).add(lab)
        dup_items = [(sha, sorted(list(labs))) for sha, labs in sha2labs.items() if len(labs) > 1]
        if dup_items:
            print("\n[dups] SHAs that appear in multiple split labels (top 50):")
            for sha, labs in dup_items[:50]:
                print(f"  {sha} -> {', '.join(labs)}")
            print(f"[dups] total multi-label SHAs in split: {len(dup_items)}")
        else:
            print("\n[dups] no multi-label SHAs found in split.")

    # 7) Show examples for forward/backward issues
    def _show_examples(tag, pairs, limit=20):
        if not pairs:
            return
        print(f"\n[{tag}] examples ({min(limit, len(pairs))} of {len(pairs)}):")
        i = 0
        for sha, lab in sorted(pairs):
            print(f"  ({lab}) {sha}")
            i += 1
            if i >= limit:
                break

    _show_examples("forward-missing base->split", missing_in_split)
    _show_examples("backward-extra split->base", extra_in_split)

    # 8) Strong multi-label validation
    ml = multilabel_recon(base_labels, split_labels_by_sha)
    print("\n[multilabel] base multi-label SHAs:", ml['counts']['base_multilabel'])
    print(f"  exact   : {ml['counts']['exact']}")
    print(f"  missing : {ml['counts']['missing']}  (split has subset of base)")
    print(f"  extra   : {ml['counts']['extra']}    (split has labels not in base)")
    print(f"  mismatch: {ml['counts']['mismatch']} (neither subset nor superset)")

    # print a few examples for each bucket
    def _print_ml_examples(name, items, maxn=10):
        if not items:
            return
        print(f"\n[multilabel:{name}] examples (up to {maxn}):")
        for i, item in enumerate(items[:maxn]):
            if name == "exact":
                sha, b = item
                print(f"  {sha}  base/split: {', '.join(b)}")
            else:
                sha, b, s = item
                print(f"  {sha}  base: {', '.join(b)}  |  split: {', '.join(s)}")

    _print_ml_examples("exact", ml['examples']['exact'])
    _print_ml_examples("missing", ml['examples']['missing'])
    _print_ml_examples("extra", ml['examples']['extra'])
    _print_ml_examples("mismatch", ml['examples']['mismatch'])

    # 9) '__labels_all' audit checks (if split was produced with --audit_multilabel)
    if audit_alllabels_by_sha or audit_alllabels_conflicts:
        mismatches, missing_audit = audit_alllabels_check(base_labels, audit_alllabels_by_sha)
        if audit_alllabels_conflicts:
            print(f"\n[audit __labels_all] conflicts across folders: {len(audit_alllabels_conflicts)} (show up to 10)")
            for i, (sha, folder_label, seen_set, new_set) in enumerate(audit_alllabels_conflicts[:10]):
                print(f"  {sha} in '{folder_label}'  seen:{sorted(list(seen_set))}  new:{sorted(list(new_set))}")
        print(f"\n[audit __labels_all] mismatches vs base: {len(mismatches)} (up to 10)")
        for i, (sha, b, a) in enumerate(mismatches[:10]):
            print(f"  {sha}  base:{', '.join(b)}  audit:{', '.join(a)}")
        print(f"[audit __labels_all] missing audit for base multi-label SHAs: {len(missing_audit)} (up to 10)")
        for i, (sha, b) in enumerate(missing_audit[:10]):
            print(f"  {sha}  base:{', '.join(b)}")
    else:
        print("\n[audit __labels_all] no audit fields found in split (skipping).")

    # 10) CSV output (per-label table)
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label", "base_cnt", "split_cnt", "miss_to_split", "extra_from_split",
                        "label_mismatch", "invalid_sha", "missing_R", "coverage_percent"])
            for r in sorted(rows, key=lambda x: (-x[1], x[0])):
                lab, bc, sc, fm, be, mm, bd, mr, cov = r
                w.writerow([lab, bc, sc, fm, be, mm, bd, mr, f"{cov:.2f}"])
        print(f"\n[write] CSV summary -> {args.output}")

if __name__ == "__main__":
    main()