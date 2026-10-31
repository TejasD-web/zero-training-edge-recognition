"""
Pre-registered threshold selection for the confidence router (dev set only).
 
RULE (committed before any dev data is read; do not change after seeing results):
  Candidate thresholds G = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99].
  Pick the SMALLEST t in G such that every dev answer the edge would keep
  (joint digit confidence >= t) is correct, and at least one answer is kept.
  If no t in G qualifies, use t = 0.99.
  A missing/unparseable edge answer counts as wrong.
  The chosen t is written to config/threshold.json and then frozen for the held-out run.
 
Usage:
  # 1) read the dev set on the edge (starts/stops llama-server), apply the rule, write config + chart
  python benchmark/select_threshold.py --manifest data/own/dev/manifest.json --image-dir data/own/dev/images
  # 2) re-apply the rule to an existing probe file (no model needed)
  python benchmark/select_threshold.py --from-json results/threshold_probe_YYYYmmdd_HHMMSS.json
 
Manifest: JSON list of {"image": ..., "truth": "23", ...} ("jersey_number" also accepted).
"""
import os, sys, json, argparse
from datetime import datetime
 
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
 
GRID = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
FALLBACK = 0.99
CONFIG = os.path.join(ROOT, "config", "threshold.json")
 
 
def load_manifest(path):
    data = json.load(open(path))
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    return [{**m, "truth": str(m.get("truth", m.get("jersey_number")))} for m in data]
 
 
def probe(manifest_path, image_dir):
    """One edge read per dev image (256-token budget, logprobs on). Same call the router makes."""
    from benchmark import run_benchmark as rb
    from benchmark import logprob_probe as lp
 
    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem < 4_500_000:
        sys.exit(f"[MEM] Only {mem // 1024} MB available (need ~4.5 GB). "
                 "Drop caches first: sync && sudo sysctl -w vm.drop_caches=3")
    rows = []
    proc, _ = rb.start_vlm_server()
    try:
        for m in load_manifest(manifest_path):
            path = os.path.join(image_dir, m["image"])
            raw, toks, t = lp.query_logprobs(rb.prep_read(path))
            ok, pred = rb.score(raw, m["truth"])
            rows.append({"image": m["image"], "truth": m["truth"], "pred": pred, "raw": raw,
                         "correct": bool(ok), "conf": lp.confidence(toks)["joint"],
                         "latency_s": t, "difficulty": m.get("difficulty")})
            print(f"  {m['image']:<24} pred={str(pred):>6} truth={m['truth']:>6} "
                  f"{'✓' if ok else '✗'} conf={rows[-1]['conf']:.3f}")
    finally:
        rb.stop_vlm_server(proc)
    return rows
 
 
def sweep(rows, t):
    kept = [r for r in rows if r["pred"] is not None and r["conf"] >= t]
    kept_ok = sum(r["correct"] for r in kept)
    return {"threshold": t, "kept": len(kept), "kept_correct": kept_ok,
            "precision": round(kept_ok / len(kept), 4) if kept else None,
            "on_device_rate": round(len(kept) / len(rows), 4)}
 
 
def apply_rule(rows):
    table = [sweep(rows, t) for t in GRID]
    for s in table:
        if s["kept"] >= 1 and s["kept_correct"] == s["kept"]:
            return s["threshold"], table, "smallest grid value with 100% precision on kept answers"
    return FALLBACK, table, "no grid value reached 100% precision; fallback"
 
 
def chart(rows, chosen, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[CHART] matplotlib not installed; skipped (pip install matplotlib)")
        return None
    ts = [i / 100 for i in range(0, 100)]
    curves = [sweep(rows, t) for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    ax.plot(ts, [c["on_device_rate"] for c in curves], label="answered on-device", color="#2a9d6f", lw=2)
    ax.plot(ts, [c["precision"] if c["precision"] is not None else float("nan") for c in curves],
            label="accuracy of on-device answers", color="#3a6fd8", lw=2)
    ax.axvline(chosen, color="#444", ls="--", lw=1)
    ax.text(chosen, 0.04, f" t = {chosen}", fontsize=9)
    wrong = [r["conf"] for r in rows if not r["correct"]]
    right = [r["conf"] for r in rows if r["correct"]]
    ax.scatter(right, [1.04] * len(right), marker="|", color="#2a9d6f", s=80, label="dev read: correct")
    ax.scatter(wrong, [1.08] * len(wrong), marker="|", color="#d64545", s=80, label="dev read: wrong")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
    ax.set_xlabel("edge confidence threshold (joint digit probability)")
    ax.set_ylabel("fraction")
    ax.set_title(f"Dev set (n={len(rows)}): threshold sweep")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png)
    return out_png
 
 
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="dev manifest (runs the edge model)")
    src.add_argument("--from-json", help="existing threshold_probe JSON (no model)")
    ap.add_argument("--image-dir", help="folder with the dev images (default: manifest's folder/images)")
    ap.add_argument("--no-config", action="store_true", help="don't write config/threshold.json")
    args = ap.parse_args()
 
    os.makedirs("results", exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    if args.manifest:
        image_dir = args.image_dir or os.path.join(os.path.dirname(args.manifest), "images")
        print(f"[PROBE] {args.manifest} ({image_dir})")
        rows = probe(args.manifest, image_dir)
        src = args.manifest
    else:
        d = json.load(open(args.from_json))
        rows, src = d["rows"], d.get("manifest", args.from_json)
 
    t, table, why = apply_rule(rows)
    n, acc = len(rows), sum(r["correct"] for r in rows)
    print(f"\nDev edge accuracy (full frame): {acc}/{n}")
    print(f"{'thresh':>7}{'on-device':>11}{'kept acc':>10}")
    for s in table:
        prec = f"{s['kept_correct']}/{s['kept']}" if s["kept"] else "-"
        print(f"{s['threshold']:>7}{s['kept']:>11}{prec:>10}{'   <- chosen' if s['threshold'] == t else ''}")
    print(f"\nChosen threshold: {t}  ({why})")
 
    out = {"manifest": src, "rule": __doc__.split("RULE")[1].split("Usage:")[0].strip(),
           "grid": GRID, "chosen": t, "reason": why, "table": table, "rows": rows}
    out_json = os.path.join("results", f"threshold_probe_{stamp}.json")
    if args.manifest:
        json.dump(out, open(out_json, "w"), indent=2)
        print(f"[LOG] Wrote {out_json}")
    png = chart(rows, t, os.path.join("results", f"threshold_sweep_{stamp}.png"))
    if png:
        print(f"[CHART] Wrote {png}")
    if not args.no_config:
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        json.dump({"threshold": t, "selected_on": src, "probe": out_json if args.manifest else args.from_json,
                   "rule": "smallest of GRID with 100% precision on kept dev answers (benchmark/select_threshold.py)",
                   "selected_at": datetime.now().isoformat(timespec="seconds")},
                  open(CONFIG, "w"), indent=2)
        print(f"[CONFIG] Wrote {CONFIG} -> router uses threshold {t}. Commit it before the held-out run.")
 
 
if __name__ == "__main__":
    main()
