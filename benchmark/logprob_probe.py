"""
Day 5 probe: is the edge model's own token probability a usable tier-1 confidence signal?

Reads every DEV image once (full frame, 256-token budget) with logprobs, then simulates routing
policies using the cloud results already stored in a benchmark JSON (no new API calls).
Dev set only -- never run this on data/heldout.

Usage (fresh boot):
  python benchmark/logprob_probe.py
  python benchmark/logprob_probe.py --bench results/benchmark_20261028_203643.json
"""
import os, sys, json, math, time, base64, glob, argparse
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from benchmark import run_benchmark as rb

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

def query_logprobs(image_path, top=5):
    import requests
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    r = requests.post(rb.VLM_URL, json={
        "model": "qwen2.5-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": rb.PROMPT},
        ]}],
        "max_tokens": 16, "temperature": 0, "logprobs": True, "top_logprobs": top,
    }, timeout=120)
    t = round(time.time() - t0, 3)
    ch = r.json()["choices"][0]
    toks = (ch.get("logprobs") or {}).get("content") or []
    return ch["message"]["content"].strip(), toks, t

def confidence(toks):
    """joint = P(all answer digits), min = weakest digit, margin = top1 - top2 on the first digit."""
    digits = [t for t in toks if any(c.isdigit() for c in t.get("token", ""))]
    if not digits:
        return {"joint": 0.0, "min": 0.0, "margin": 0.0, "n_digit_tokens": 0}
    lps = [t["logprob"] for t in digits]
    tops = sorted((x["logprob"] for x in digits[0].get("top_logprobs", [])), reverse=True)
    p1 = math.exp(tops[0]) if tops else math.exp(digits[0]["logprob"])
    p2 = math.exp(tops[1]) if len(tops) > 1 else 0.0
    return {"joint": round(math.exp(sum(lps)), 4), "min": round(math.exp(min(lps)), 4),
            "margin": round(p1 - p2, 4), "n_digit_tokens": len(digits)}

def bench_pred(brow, key):
    v = (brow or {}).get(key)
    return v.get("pred") if v else None

def bench_ok(brow, key):
    v = (brow or {}).get(key)
    return bool(v and v.get("correct"))

def simulate(rows, bench, signal, t, use_sam3):
    """Tier 1: keep edge answer if signal >= t. Optional tier 2: keep if SAM 3.1 crop read agrees.
    Else Muse Spark (full image) from the stored benchmark."""
    ok = kept = kept_ok = sam = muse = 0
    for r in rows:
        b = bench.get(r["image"])
        if r["conf"][signal] >= t:
            kept += 1
            kept_ok += r["correct"]
            ok += r["correct"]
            continue
        if use_sam3:
            sam += 1
            if r["pred"] is not None and bench_pred(b, "cloud_sam_edge_read") == r["pred"]:
                ok += r["correct"]
                continue
        muse += 1
        ok += bench_ok(b, "muse_full")
    return ok, kept, kept_ok, sam, muse

def agreement_cascade(bench, names):
    """The current route.py policy, recomputed from the stored benchmark for comparison."""
    ok = kept = sam = muse = 0
    for n in names:
        b = bench.get(n)
        full = bench_pred(b, "full_image")
        if full is not None and full == bench_pred(b, "vlm_box"):
            kept += 1; ok += bench_ok(b, "full_image"); continue
        sam += 1
        if full is not None and full == bench_pred(b, "cloud_sam_edge_read"):
            ok += bench_ok(b, "full_image"); continue
        muse += 1; ok += bench_ok(b, "muse_full")
    return ok, kept, sam, muse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", help="benchmark JSON with --cloud results (default: latest that has muse_full)")
    args = ap.parse_args()

    bench_path = args.bench
    if not bench_path:
        for p in sorted(glob.glob("results/benchmark_*.json"), reverse=True):
            if any("muse_full" in r for r in json.load(open(p))["rows"]):
                bench_path = p
                break
    bench = {r["image"]: r for r in json.load(open(bench_path))["rows"]}
    print(f"[BENCH] cloud results from {bench_path}")

    mem_avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem_avail < 4_500_000:
        sys.exit(f"[MEM] Only {mem_avail // 1024} MB available (need ~4.5 GB). Drop caches first: sync && sudo sysctl -w vm.drop_caches=3")

    manifest = rb.load_manifest()
    proc, _ = rb.start_vlm_server()
    rows = []
    try:
        for m in manifest:
            path = os.path.join(rb.IMAGE_DIR, m["image"])
            raw, toks, t = query_logprobs(rb.prep_read(path))
            if not toks and not rows:
                print("[WARN] response had no logprobs.content -- check llama-server version/response format")
            ok, pred = rb.score(raw, m["jersey_number"])
            rows.append({"image": m["image"], "truth": m["jersey_number"], "difficulty": m.get("difficulty"),
                         "pred": pred, "raw": raw, "correct": ok, "latency_s": t, "conf": confidence(toks),
                         "tokens": [{"token": x.get("token"), "logprob": x.get("logprob")} for x in toks]})
    finally:
        rb.stop_vlm_server(proc)

    print(f"\n{'image':<12}{'pred':>6}{'truth':>10}  ok  {'joint':>6}{'min':>7}{'margin':>8}  difficulty")
    for r in sorted(rows, key=lambda r: -r["conf"]["joint"]):
        c = r["conf"]
        print(f"{r['image']:<12}{str(r['pred']):>6}{r['truth']:>10}  {'✓' if r['correct'] else '✗'}  "
              f"{c['joint']:>6.3f}{c['min']:>7.3f}{c['margin']:>8.3f}  {r['difficulty']}")

    n = len(rows)
    acc = sum(r["correct"] for r in rows)
    print(f"\nEdge full-image accuracy this run: {acc}/{n}  (mean read {sum(r['latency_s'] for r in rows) / n:.1f}s)")

    a_ok, a_kept, a_sam, a_muse = agreement_cascade(bench, [r["image"] for r in rows])
    print(f"\nCurrent agreement cascade (from stored benchmark): {a_ok}/{n} correct, "
          f"{a_kept} on-device, {a_sam} SAM 3.1, {a_muse} Muse  (tier 1 costs ~14s/frame)")

    summary = []
    for use_sam3 in (False, True):
        label = "logprob -> SAM 3.1 agree -> Muse" if use_sam3 else "logprob -> Muse"
        print(f"\nPolicy: {label}   (tier 1 costs ~2.6s/frame)")
        print(f"  {'signal':<7}{'thresh':>7}{'acc':>8}{'on-device':>11}{'kept prec':>11}{'SAM3':>6}{'Muse':>6}")
        for signal in ("joint", "min", "margin"):
            for t in THRESHOLDS:
                ok, kept, kept_ok, sam, muse = simulate(rows, bench, signal, t, use_sam3)
                prec = f"{kept_ok}/{kept}" if kept else "-"
                print(f"  {signal:<7}{t:>7}{f'{ok}/{n}':>8}{kept:>11}{prec:>11}{sam:>6}{muse:>6}")
                summary.append({"policy": label, "signal": signal, "threshold": t, "correct": ok, "n": n,
                                "on_device": kept, "kept_correct": kept_ok, "sam3_calls": sam, "muse_calls": muse})

    out = os.path.join(rb.RESULTS_DIR, f"logprob_probe_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump({"bench": bench_path, "rows": rows, "policies": summary,
                   "agreement_cascade": {"correct": a_ok, "on_device": a_kept, "sam3_calls": a_sam, "muse_calls": a_muse}},
                  f, indent=2)
    print(f"\n[LOG] Wrote {out}")
    print("Threshold is chosen on this DEV set; the held-out run then uses that fixed value.")

if __name__ == "__main__":
    main()
