"""
Edge→cloud router for fine-grained reading (jersey numbers as the benchmark).

Policy "confidence" (default, pre-registered from benchmark/logprob_probe.py on the dev set):
  Tier 1  on-device : edge Qwen reads the full frame; answer if joint digit probability >= threshold (0.9).
  Tier 3  cloud read: otherwise Muse Spark reads the full frame.

Policy "agreement" (original cascade, kept for comparison):
  Tier 1  on-device   : edge full-frame read + Qwen-grounded crop read agree → answer.
  Tier 2  cloud locate: SAM 3.1 crop read on edge agrees with the full-frame read → answer.
  Tier 3  cloud read  : Muse Spark reads the full frame.

Usage (fresh boot; the script starts and stops llama-server itself):
  python router/route.py data/images/img_001.jpg [more images ...]
  python router/route.py --manifest                          # dev set, confidence policy
  python router/route.py --manifest --policy agreement       # original cascade
  python router/route.py --manifest --max-tier 1             # offline mode: never call the cloud
  python router/route.py --manifest data/heldout/manifest.json --image-dir data/heldout/images

Read now, verify later (confidence policy):
  Low-confidence frames are QUEUED (edge answer kept, marked unverified) when running offline
  (--max-tier 1) or when the Meta Model API is unreachable. Later, with a connection:
  python router/route.py --sync           # send only queued frames to Muse Spark; confirm or correct
  python router/route.py --clear-queue    # reset the queue (e.g. before a demo)
"""
import os, sys, json, time, argparse, shutil, socket
from datetime import datetime
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from benchmark import run_benchmark as rb   # edge server, reads, grounding, crops, scoring
from router import meta_client as mc        # SAM 3.1 + Muse Spark
from benchmark import logprob_probe as lp    # logprob query + confidence

def _load_threshold(default=0.9):
    """Threshold chosen on the dev set by benchmark/select_threshold.py; frozen for held-out runs."""
    try:
        return float(json.load(open(os.path.join(ROOT, "config", "threshold.json")))["threshold"])
    except (OSError, KeyError, ValueError):
        return default

CONF_THRESHOLD = _load_threshold()
QUEUE_DIR = os.path.join(rb.RESULTS_DIR, "queue")
QUEUE_FILE = os.path.join(QUEUE_DIR, "queue.jsonl")

def cloud_available(host="api.meta.ai", port=443, timeout=2.0):
    """Fast reachability check so a dropped link costs ~2s, not a 90s request timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def enqueue(image_path, rec, truth=None):
    os.makedirs(os.path.join(QUEUE_DIR, "images"), exist_ok=True)
    dst = os.path.join(QUEUE_DIR, "images", rec["image"])
    shutil.copy2(image_path, dst)
    entry = {"image": rec["image"], "path": dst, "edge_answer": rec["answer"],
             "edge_conf": (rec.get("edge_conf") or {}).get("joint"), "truth": truth,
             "bytes": os.path.getsize(dst), "status": "unverified",
             "queued_at": datetime.now().isoformat(timespec="seconds")}
    with open(QUEUE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    rec["queued"] = True

def sync():
    """Send only queued (unverified) frames to Muse Spark; confirm or correct the edge answers."""
    if not os.path.exists(QUEUE_FILE):
        print("[SYNC] Queue is empty.")
        return
    entries = [json.loads(l) for l in open(QUEUE_FILE) if l.strip()]
    pending = [e for e in entries if e["status"] == "unverified"]
    if not pending:
        print(f"[SYNC] Nothing pending ({len(entries)} already verified).")
        return
    if not cloud_available():
        sys.exit(f"[SYNC] Meta Model API unreachable; {len(pending)} frames stay queued. Try again later.")
    print(f"[SYNC] Connection OK. Verifying {len(pending)} queued frames with Muse Spark...")
    up_bytes = corrected = 0
    for e in pending:
        try:
            raw, t = mc.muse_read(e["path"])
        except Exception as ex:
            print(f"[SYNC] {e['image']}: ERROR {str(ex)[:120]} (stays queued)")
            continue
        ans = rb.normalize(raw)
        e.update({"status": "verified", "cloud_answer": ans, "cloud_latency_s": t,
                  "corrected": ans != e["edge_answer"], "verified_at": datetime.now().isoformat(timespec="seconds")})
        up_bytes += e["bytes"]
        corrected += e["corrected"]
        mark = ""
        if e.get("truth"):
            e["edge_correct"] = rb.score(e["edge_answer"] or "", e["truth"])[0]
            e["correct"] = rb.score(ans or "", e["truth"])[0]
            mark = f"  truth={e['truth']} edge {'✓' if e['edge_correct'] else '✗'} -> final {'✓' if e['correct'] else '✗'}"
        verb = ("FIXED" if e.get("edge_correct") is False else "changed") if e["corrected"] else "confirmed"
        print(f"[SYNC] {e['image']}: edge {e['edge_answer']} (conf {e['edge_conf']}) -> Muse {ans}  {verb}  {t}s{mark}")
    with open(QUEUE_FILE, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    done_now = [e for e in pending if e["status"] == "verified"]
    print(f"\n{'=' * 60}\nSYNC SUMMARY")
    print(f"  verified           {len(done_now)}/{len(pending)} queued frames")
    print(f"  edge confirmed     {len(done_now) - corrected}")
    print(f"  answer changed     {corrected}")
    fixed = sum(1 for e in done_now if e.get("edge_correct") is False and e.get("correct"))
    if any("correct" in e for e in done_now):
        print(f"  edge errors fixed  {fixed}")
    print(f"  uploaded           {up_bytes / 1e6:.2f} MB ({len(done_now)} images)")
    scored = [e for e in done_now if "correct" in e]
    if scored:
        print(f"  accuracy on queued frames: edge {sum(e['edge_correct'] for e in scored)}/{len(scored)}"
              f" -> after sync {sum(e['correct'] for e in scored)}/{len(scored)}")
    print("=" * 60)

def edge_read(image_path):
    raw, t = rb.query_vlm(rb.prep_read(image_path))
    return rb.normalize(raw), raw, t

def new_record(image_path):
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

    return rec, step, done

def muse_tier(image_path, rec, step, done, fallback):
    try:
        raw, t = mc.muse_read(image_path)
        rec["muse_calls"] += 1
        p_muse = rb.normalize(raw)
        step("muse_full", p_muse, t, raw=raw)
        return done(3, p_muse, "escalated to Muse Spark")
    except Exception as e:
        step("muse_full", None, None, error=str(e)[:200])
        rec["needs_queue"] = True
        return done(3, fallback, "LOW CONFIDENCE: Muse call failed; edge answer kept, queued for verification")

def route_confidence(image_path, max_tier=3, threshold=CONF_THRESHOLD):
    """Tier 1: edge read with its own token probability. Confident → answer. Else → Muse Spark."""
    rec, step, done = new_record(image_path)
    raw, toks, t = lp.query_logprobs(rb.prep_read(image_path))
    p_full, conf = rb.normalize(raw), lp.confidence(toks)
    step("edge_full", p_full, t, raw=raw, conf=conf["joint"])
    rec["edge_conf"] = conf
    if p_full is not None and conf["joint"] >= threshold:
        return done(1, p_full, f"edge confidence {conf['joint']:.3f} >= {threshold}")
    if max_tier < 3:
        rec["needs_queue"] = True
        return done(1, p_full, f"LOW CONFIDENCE: edge confidence {conf['joint']:.3f} < {threshold}, offline; queued for verification")
    if not cloud_available():
        rec["needs_queue"] = True
        return done(1, p_full, f"LOW CONFIDENCE: edge confidence {conf['joint']:.3f} < {threshold}, no connection; queued for verification")
    return muse_tier(image_path, rec, step, done, p_full)

def route_agreement(image_path, max_tier=3):
    """Original cascade: two edge views agree → SAM 3.1 crop agrees → Muse Spark."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    rec, step, done = new_record(image_path)

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
    return muse_tier(image_path, rec, step, done, p_full)

def route(image_path, max_tier=3, policy="confidence"):
    if policy == "agreement":
        return route_agreement(image_path, max_tier)
    return route_confidence(image_path, max_tier)

def print_record(rec, truth=None):
    mark = ""
    if truth is not None:
        ok, _ = rb.score(rec["answer"] or "", truth)
        rec["truth"], rec["correct"] = truth, ok
        mark = f"  truth={truth} {'✓' if ok else '✗'}"
    where = "on-device" if not rec["left_device"] else f"cloud: {rec['sam3_calls']} SAM 3.1, {rec['muse_calls']} Muse"
    print(f"[ROUTE] {rec['image']}: answer={rec['answer']}  tier {rec['tier']} ({where})  {rec['total_s']}s{mark}")
    print(f"        {rec['reason']}" + ("  [QUEUED]" if rec.get("queued") else ""))
    for s in rec["steps"]:
        bits = [f"pred={s['pred']}" if s.get("pred") is not None else "",
                f"box={s['box']}" if s.get("box") else "",
                f"n_boxes={s['n_boxes']}" if "n_boxes" in s else "",
                f"conf={s['conf']:.3f}" if "conf" in s else "",
                f"ERROR {s['error']}" if "error" in s else ""]
        lat = f"{s['latency_s']}s" if s.get("latency_s") is not None else "-"
        print(f"          {s['step']:<20} {lat:>7}  {' '.join(b for b in bits if b)}")

def summarize(records):
    n = len(records)
    scored = [r for r in records if "correct" in r]
    tiers = {t: sum(r["tier"] == t for r in records) for t in (1, 2, 3)}
    lat = sorted(r["total_s"] for r in records)
    print(f"\n{'=' * 60}\nROUTER SUMMARY ({n} frames, policy={records[0].get('policy', '?')})")
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
    q = sum(bool(r.get("queued")) for r in records)
    if q:
        print(f"  queued for sync    {q}  (run: python router/route.py --sync)")
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
    ap.add_argument("--policy", default="confidence", choices=["confidence", "agreement"])
    ap.add_argument("--sync", action="store_true", help="verify queued frames with Muse Spark (no edge server needed)")
    ap.add_argument("--clear-queue", action="store_true", help="delete the offline queue")
    args = ap.parse_args()

    if args.clear_queue:
        shutil.rmtree(QUEUE_DIR, ignore_errors=True)
        print("[QUEUE] cleared")
        return
    if args.sync:
        sync()
        return

    jobs = []
    if args.manifest:
        with open(args.manifest) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next(v for v in data.values() if isinstance(v, list))
        jobs = [(os.path.join(args.image_dir, m["image"]), m.get("truth", m.get("jersey_number"))) for m in data]
    jobs += [(p, None) for p in args.images]
    if not jobs:
        ap.error("give image paths or --manifest")

    mem_avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem_avail < 4_500_000:
        sys.exit(f"[MEM] Only {mem_avail // 1024} MB available (need ~4.5 GB). Drop caches first: sync && sudo sysctl -w vm.drop_caches=3")

    proc, _ = rb.start_vlm_server()
    records = []
    try:
        for path, truth in jobs:
            if proc.poll() is not None:
                print("[VLM] Server died — see results/vlm_server.log")
                break
            rec = route(path, max_tier=args.max_tier, policy=args.policy)
            rec["policy"] = args.policy
            if rec.get("needs_queue"):
                enqueue(path, rec, truth)
            print_record(rec, truth)
            records.append(rec)
    finally:
        rb.stop_vlm_server(proc)

    if records:
        summarize(records)
        out = os.path.join(rb.RESULTS_DIR, f"route_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(out, "w") as f:
            json.dump({"policy": args.policy, "threshold": CONF_THRESHOLD if args.policy == "confidence" else None,
                       "max_tier": args.max_tier, "manifest": args.manifest, "records": records}, f, indent=2)
        print(f"[LOG] Wrote {out}")

if __name__ == "__main__":
    main()
