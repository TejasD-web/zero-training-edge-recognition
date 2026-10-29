"""
Routing analysis: accuracy vs. cloud usage for edge/cloud policies, computed from a benchmark JSON.
Usage:
  python benchmark/analyze_routing.py                         # latest results/benchmark_*.json
  python benchmark/analyze_routing.py results/benchmark_X.json
  python benchmark/analyze_routing.py --md results/routing.md # also write the tables as markdown

Policy "agree(full, X)": answer on the edge with the full-image read when it matches view X,
otherwise escalate the frame to Muse Spark (full image). A missing view counts as disagreement.
"""
import sys, json, glob, os, argparse

EDGE = "full_image"
CLOUD = "muse_full"
CONDITIONS = [
    ("full_image", "Edge Qwen, full image"),
    ("sam_crop", "Edge Qwen, center-point SAM 2.1 crop"),
    ("vlm_box", "Edge Qwen, Qwen-grounded box crop"),
    ("vlm_box_sam", "Edge Qwen, Qwen box -> SAM 2.1 crop"),
    ("cloud_sam_edge_read", "SAM 3.1 crop -> edge Qwen read"),
    ("cloud_sam_muse", "SAM 3.1 crop -> Muse Spark read"),
    ("muse_full", "Muse Spark, full image"),
]
# (partner views, label, extra cloud calls per frame besides Muse escalations)
POLICIES = [
    (["sam_crop"], "agree(full, center SAM 2.1 crop)", None),
    (["vlm_box"], "agree(full, Qwen box crop)", None),
    (["vlm_box_sam"], "agree(full, Qwen box -> SAM 2.1)", None),
    (["sam_crop", "vlm_box", "vlm_box_sam"], "agree(full, any edge crop)", None),
    (["cloud_sam_edge_read"], "agree(full, SAM 3.1 crop -> edge read)", "SAM 3.1 on every frame"),
]

def pred(row, key):
    v = row.get(key)
    return v.get("pred") if v else None

def correct(row, key):
    v = row.get(key)
    return bool(v and v.get("correct"))

def condition_table(rows):
    n = len(rows)
    out = []
    for key, label in CONDITIONS:
        if not any(key in r for r in rows):
            continue
        c = sum(correct(r, key) for r in rows)
        lat = [r[key]["vlm_time_s"] for r in rows if key in r and "vlm_time_s" in r[key]]
        out.append((label, c, n, sum(lat) / len(lat) if lat else None))
    return out

def run_policy(rows, partners):
    kept = kept_ok = esc = esc_ok = 0
    detail = []
    for r in rows:
        p_full = pred(r, EDGE)
        agrees = p_full is not None and any(pred(r, k) == p_full for k in partners)
        if agrees:
            kept += 1
            ok = correct(r, EDGE)
            kept_ok += ok
            detail.append((r["image"], "edge", p_full, ok))
        else:
            esc += 1
            ok = correct(r, CLOUD)
            esc_ok += ok
            detail.append((r["image"], "cloud", pred(r, CLOUD), ok))
    return kept, kept_ok, esc, esc_ok, detail

def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--md", help="write markdown tables to this file")
    ap.add_argument("--detail", action="store_true", help="print per-image routing decisions")
    args = ap.parse_args()

    path = args.path or sorted(glob.glob("results/benchmark_*.json"))[-1]
    data = json.load(open(path))
    rows = data["rows"]
    n = len(rows)
    has_cloud = any(CLOUD in r for r in rows)

    lines = [f"Source: `{os.path.basename(path)}` ({n} frames)", ""]

    lines += ["| Condition | Accuracy | Mean read latency |", "|---|---|---|"]
    for label, c, tot, lat in condition_table(rows):
        lines.append(f"| {label} | {pct(c, tot)} ({c}/{tot}) | {f'{lat:.1f}s' if lat else '-'} |")

    edge_only = ["full_image", "sam_crop", "vlm_box", "vlm_box_sam"]
    any_ok = sum(any(correct(r, k) for k in edge_only) for r in rows)
    lines += ["", f"Edge-only oracle (at least one fully on-device view correct): {pct(any_ok, n)} ({any_ok}/{n})", ""]

    details = {}
    if not has_cloud:
        lines.append("No `muse_full` results in this run; routing policies need `--cloud`.")
    else:
        edge_ok = sum(correct(r, EDGE) for r in rows)
        cloud_ok = sum(correct(r, CLOUD) for r in rows)
        lines += ["| Policy | Accuracy | Frames sent to Muse | Edge-kept precision | Other cloud calls |",
                  "|---|---|---|---|---|",
                  f"| Edge only | {pct(edge_ok, n)} ({edge_ok}/{n}) | 0% (0/{n}) | - | - |"]
        for partners, label, extra in POLICIES:
            if not any(k in r for r in rows for k in partners):
                continue
            kept, kept_ok, esc, esc_ok, detail = run_policy(rows, partners)
            details[label] = detail
            acc = kept_ok + esc_ok
            lines.append(f"| {label} | {pct(acc, n)} ({acc}/{n}) | {pct(esc, n)} ({esc}/{n}) | "
                         f"{pct(kept_ok, kept)} ({kept_ok}/{kept}) | {extra or '-'} |")
        lines.append(f"| Cloud only | {pct(cloud_ok, n)} ({cloud_ok}/{n}) | 100% ({n}/{n}) | - | - |")
        lines += ["", "Edge-kept precision = accuracy on the frames the policy answered on the edge.",
                  "Policies were chosen on this set; confirm on a held-out set before reporting."]

    text = "\n".join(lines)
    print(text)

    if args.detail:
        for label, detail in details.items():
            print(f"\n{label}")
            for img, where, p, ok in detail:
                print(f"  {img}: {where:5s} pred={p} {'✓' if ok else '✗'}")

    if args.md:
        with open(args.md, "w") as f:
            f.write(text + "\n")
        print(f"\n[LOG] Wrote {args.md}")

if __name__ == "__main__":
    main()
