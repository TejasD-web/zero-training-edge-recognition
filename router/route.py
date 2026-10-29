"""
Edge→cloud cascade router for fine-grained reading (jersey numbers as the benchmark).

  Tier 1  on-device   : edge Qwen reads the full frame + a Qwen-grounded crop. Agree → answer.
  Tier 2  cloud locate: SAM 3.1 finds "jersey number", edge Qwen reads that crop. Agrees with tier-1 full read → answer.
  Tier 3  cloud read  : Muse Spark reads the full frame.

Usage (fresh boot; the script starts and stops llama-server itself):
  python router/route.py data/images/img_001.jpg [more images ...]
  python router/route.py --manifest                 # every image in data/manifest.json, scored
  python router/route.py --manifest --max-tier 1    # offline mode: never call the cloud
"""
import os, sys, json, time, argparse
from datetime import datetime
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from benchmark import run_benchmark as rb   # edge server, reads, grounding, crops, scoring
from router import meta_client as mc        # SAM 3.1 + Muse Spark

def edge_read(image_path):
    raw, t = rb.query_vlm(rb.prep_read(image_path))
    return rb.normalize(raw), raw, t

def route(image_path, max_tier=3):
    """Run the cascade on one image. Returns a decision record."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    rec = {"image": os.path.basename(image_path), "steps": [], "sam3_calls": 0, "muse_calls": 0}
    t_start = time.time()

    def step(name, pred, latency, **extra):
        rec["steps"].append({"step": name, "pred": pred, "latency_s": latency, **extra})

    def done(tier, answer, reason):
        rec.update({"tier": tier, "answer": answer, "reason": reason,
                    "confident": not reason.startswith("LOW CONFIDENCE"),
                    "left_device": rec["sam3_calls"] + rec["muse_calls"] > 0,
                    "total_s": round(time.time() - t_start, 3)})
        return rec

    # ── Tier 1: fully on-device ──
    p_full, raw, t = edge_read(image_path)
    step("edge_full", p_full, t, raw=raw)

    box, graw, tg = rb.ground(image_path)
    step("edge_ground", None, tg, box=box)
    p_box = None
    if box:
        crop_path, _ = rb.box_crop(image_path, box)
        p_box, raw, t = edge_read(crop_path)
        step("edge_box_read", p_box, t, raw=raw)

    if p_full is not None and p_full == p_box:
        return done(1, p_full, "edge full-frame and grounded-crop reads agree")
    if max_tier < 2:
        return done(1, p_full, "LOW CONFIDENCE: views disagree, cloud disabled (offline mode)")

    # ── Tier 2: cloud localizes, edge reads ──
    try:
        boxes, _, ts = mc.sam_segment(image_path)
        rec["sam3_calls"] += 1
        step("sam3_locate", None, ts, n_boxes=len(boxes))
    except Exception as e:
        boxes = []
        step("sam3_locate", None, None, error=str(e)[:200])

    if boxes:
        big = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        crop = mc.pad_crop(Image.open(image_path).convert("RGB"), big)
        crop_path = os.path.join(rb.CROP_DIR, f"route_sam3_{stem}.jpg")
        crop.save(crop_path, quality=95)
        p_c2e, raw, t = edge_read(crop_path)
        step("edge_read_sam3_crop", p_c2e, t, raw=raw, box=big)
        if p_full is not None and p_c2e == p_full:
            return done(2, p_full, "SAM 3.1 crop read on edge agrees with full-frame read")

    if max_tier < 3:
        return done(2, p_full, "LOW CONFIDENCE: no agreement, Muse disabled")

    # ── Tier 3: cloud reads ──
    try:
        raw, t = mc.muse_read(image_path)
        rec["muse_calls"] += 1
        p_muse = rb.normalize(raw)
        step("muse_full", p_muse, t, raw=raw)
        return done(3, p_muse, "escalated: edge views disagree")
    except Exception as e:
        step("muse_full", None, None, error=str(e)[:200])
        return done(3, p_full, "LOW CONFIDENCE: Muse call failed, falling back to edge full-frame read")

def print_record(rec, truth=None):
    mark = ""
    if truth is not None:
        ok, _ = rb.score(rec["answer"] or "", truth)
        rec["truth"], rec["correct"] = truth, ok
        mark = f"  truth={truth} {'✓' if ok else '✗'}"
    where = "on-device" if not rec["left_device"] else f"cloud: {rec['sam3_calls']} SAM 3.1, {rec['muse_calls']} Muse"
    print(f"[ROUTE] {rec['image']}: answer={rec['answer']}  tier {rec['tier']} ({where})  {rec['total_s']}s{mark}")
    print(f"        {rec['reason']}")
    for s in rec["steps"]:
        bits = [f"pred={s['pred']}" if s.get("pred") is not None else "",
                f"box={s['box']}" if s.get("box") else "",
                f"n_boxes={s['n_boxes']}" if "n_boxes" in s else "",
                f"ERROR {s['error']}" if "error" in s else ""]
        lat = f"{s['latency_s']}s" if s.get("latency_s") is not None else "-"
        print(f"          {s['step']:<20} {lat:>7}  {' '.join(b for b in bits if b)}")

def summarize(records):
    n = len(records)
    scored = [r for r in records if "correct" in r]
    tiers = {t: sum(r["tier"] == t for r in records) for t in (1, 2, 3)}
    lat = sorted(r["total_s"] for r in records)
    print(f"\n{'=' * 60}\nROUTER SUMMARY ({n} frames)")
    if scored:
        c = sum(r["correct"] for r in scored)
        print(f"  accuracy           {c}/{len(scored)} ({100 * c / len(scored):.0f}%)")
    print(f"  answered at tier   1: {tiers[1]}  2: {tiers[2]}  3: {tiers[3]}")
    low = [r for r in records if not r["confident"]]
    if low:
        lc = [r for r in low if "correct" in r]
        acc = f" ({sum(r['correct'] for r in lc)}/{len(lc)} correct)" if lc else ""
        print(f"  low-confidence     {len(low)}/{n}{acc}")
    print(f"  stayed on-device   {sum(not r['left_device'] for r in records)}/{n}")
    print(f"  SAM 3.1 calls      {sum(r['sam3_calls'] for r in records)}")
    print(f"  Muse Spark calls   {sum(r['muse_calls'] for r in records)}")
    print(f"  latency            mean {sum(lat) / n:.1f}s  p50 {lat[n // 2]:.1f}s  max {lat[-1]:.1f}s")
    print("=" * 60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--manifest", nargs="?", const=rb.MANIFEST, help="route + score every image in a manifest")
    ap.add_argument("--image-dir", default=rb.IMAGE_DIR)
    ap.add_argument("--max-tier", type=int, default=3, choices=[1, 2, 3], help="1 = offline, 2 = no Muse")
    args = ap.parse_args()

    jobs = []
    if args.manifest:
        with open(args.manifest) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next(v for v in data.values() if isinstance(v, list))
        jobs = [(os.path.join(args.image_dir, m["image"]), m.get("jersey_number")) for m in data]
    jobs += [(p, None) for p in args.images]
    if not jobs:
        ap.error("give image paths or --manifest")

    cma_free = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("CmaFree"))
    if cma_free < 900_000:
        sys.exit(f"[CMA] Only {cma_free // 1024} MB free (need ~900+). Reboot first: sudo reboot")

    proc, _ = rb.start_vlm_server()
    records = []
    try:
        for path, truth in jobs:
            if proc.poll() is not None:
                print("[VLM] Server died — see results/vlm_server.log")
                break
            rec = route(path, max_tier=args.max_tier)
            print_record(rec, truth)
            records.append(rec)
    finally:
        rb.stop_vlm_server(proc)

    if records:
        summarize(records)
        out = os.path.join(rb.RESULTS_DIR, f"route_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(out, "w") as f:
            json.dump({"max_tier": args.max_tier, "records": records}, f, indent=2)
        print(f"[LOG] Wrote {out}")

if __name__ == "__main__":
    main()
