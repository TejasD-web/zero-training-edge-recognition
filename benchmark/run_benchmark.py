"""
Benchmark harness: VLM server (first, on clean CMA) → [grounding] → SAM batch (one process) → score.
Usage:
  python benchmark/run_benchmark.py                          # center-point SAM crop only
  python benchmark/run_benchmark.py --baseline               # + full image (no SAM)
  python benchmark/run_benchmark.py --baseline --grounding   # + Qwen box crop, + Qwen box → SAM crop
Reboot before each full run so CMA starts clean (r36.4.7 leaks CMA per process exit).
"""
import sys, os, re, json, time, base64, subprocess, statistics, argparse
from datetime import datetime
from PIL import Image

# === CONFIG ===
IMAGE_DIR = "data/images"
MANIFEST = "data/manifest.json"
CROP_DIR = "results/crops"
RESULTS_DIR = "results"
LLAMA_SERVER = "/mnt/nvme/project/llama.cpp/build/bin/llama-server"
VLM_MODEL = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
VLM_MMPROJ = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf"
VLM_URL = "http://localhost:8080/v1/chat/completions"
SAM_DEVICE = "cpu"   # "cpu" = fallback if SAM can't share GPU with the server
VLM_NGL = "5"        # 99 starves the 433 MB vision buffer of CMA
PROMPT = "What jersey number is the player wearing? Reply with ONLY the number, nothing else."
GROUND_PROMPT = ("Locate the jersey number of the player whose number is most clearly visible. "
                 'Output its bounding box as JSON: {"bbox_2d": [x1, y1, x2, y2]}')
GROUND_TARGET_TOKENS = 1000   # Qwen-VL needs ~1024 image tokens for reliable grounding

os.makedirs(CROP_DIR, exist_ok=True)

# ── SAM batch worker: one model load; per image a center-point crop and (if given) a box-prompt crop ──

SAM_BATCH_SCRIPT = r'''
import sys, time, json, os, numpy as np, torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

jobs = json.load(open(sys.argv[1]))
crop_dir, device = sys.argv[2], sys.argv[3]
MAX_ATTEMPTS = 3

t0 = time.time()
model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml",
                   "/mnt/nvme/models/sam2/sam2.1_hiera_tiny.pt", device=device)
predictor = SAM2ImagePredictor(model)
print(json.dumps({"event": "loaded", "sam_load_time": round(time.time() - t0, 3)}), flush=True)

# Warm-up: absorb the first-allocation NVML assert on a dummy image
for _ in range(5):
    try:
        predictor.set_image(np.zeros((1024, 1024, 3), dtype=np.uint8))
        break
    except RuntimeError:
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(0.5)

def mask_crop(image_np, mask, union_box=None, pad=0.1):
    h, w = image_np.shape[:2]
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if rows.any() and cols.any():
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
    else:
        x0, y0, x1, y1 = union_box if union_box else (0, 0, w - 1, h - 1)
    if union_box:  # never crop tighter than the grounding box
        bx0, by0, bx1, by1 = union_box
        x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    y0, y1 = max(0, int(y0) - py), min(h, int(y1) + py)
    x0, x1 = max(0, int(x0) - px), min(w, int(x1) + px)
    return image_np[y0:y1, x0:x1]

def with_retry(fn):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(), attempt
        except RuntimeError:
            if device == "cuda":
                torch.cuda.empty_cache()
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(0.5)

for job in jobs:
    path, box = job["path"], job.get("box")
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    try:
        image_np = np.array(Image.open(path).convert("RGB"))
        h, w = image_np.shape[:2]

        def point_pass():
            t1 = time.time()
            predictor.set_image(image_np)
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[w // 2, h // 2]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            return masks, scores, time.time() - t1
        (masks, scores, seg_time), attempts = with_retry(point_pass)

        crop = mask_crop(image_np, masks[int(np.argmax(scores))])
        crop_path = os.path.join(crop_dir, f"crop_{stem}.jpg")
        Image.fromarray(crop).save(crop_path)
        ev = {"event": "crop", "image": name, "crop_path": crop_path,
              "crop_size": f"{crop.shape[1]}x{crop.shape[0]}", "image_size": f"{w}x{h}",
              "sam_seg_time": round(seg_time, 3), "attempts": attempts,
              "sam_scores": [round(float(s), 4) for s in scores]}

        if box:
            def box_pass():
                t2 = time.time()
                bmasks, bscores, _ = predictor.predict(box=np.array(box, dtype=np.float32),
                                                       multimask_output=False)
                return bmasks, bscores, time.time() - t2
            (bmasks, bscores, bseg), _ = with_retry(box_pass)
            bcrop = mask_crop(image_np, bmasks[0], union_box=box)
            bcrop_path = os.path.join(crop_dir, f"crop_vsam_{stem}.jpg")
            Image.fromarray(bcrop).save(bcrop_path)
            ev.update({"box_crop_path": bcrop_path,
                       "box_crop_size": f"{bcrop.shape[1]}x{bcrop.shape[0]}",
                       "box_seg_time": round(bseg, 3), "box_score": round(float(bscores[0]), 4)})
        print(json.dumps(ev), flush=True)
    except Exception as e:
        print(json.dumps({"event": "error", "image": name, "error": str(e)[:200]}), flush=True)
'''

def run_sam_batch(jobs):
    print(f"[SAM] Batch segmenting {len(jobs)} images (device={SAM_DEVICE})...")
    jobs_path = os.path.join(RESULTS_DIR, "sam_jobs.json")
    with open(jobs_path, "w") as f:
        json.dump(jobs, f)
    proc = subprocess.run(
        [sys.executable, "-c", SAM_BATCH_SCRIPT, jobs_path, CROP_DIR, SAM_DEVICE],
        capture_output=True, text=True,
        env={**os.environ, **({"CUDA_VISIBLE_DEVICES": ""} if SAM_DEVICE == "cpu" else {})},
    )
    crops, load_time = {}, None
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev["event"] == "loaded":
            load_time = ev["sam_load_time"]
            print(f"[SAM] Model loaded in {load_time}s")
        elif ev["event"] == "crop":
            crops[ev["image"]] = ev
            retry = f" (attempt {ev['attempts']})" if ev["attempts"] > 1 else ""
            extra = f" | box→SAM {ev['box_crop_size']}" if "box_crop_path" in ev else ""
            print(f"[SAM] {ev['image']}: {ev['crop_size']} in {ev['sam_seg_time']}s{retry}{extra}")
        elif ev["event"] == "error":
            print(f"[SAM] {ev['image']}: ERROR {ev['error']}")
    if proc.returncode != 0:
        print(f"[SAM] Process exited with code {proc.returncode}:\n{proc.stderr[-2000:]}")
    return crops, load_time

# ── VLM server (one session for all queries) ──

def start_vlm_server():
    print(f"[VLM] Starting llama-server (ngl={VLM_NGL}, unified memory)...")
    log_file = open(os.path.join(RESULTS_DIR, "vlm_server.log"), "w")
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", VLM_MODEL, "--mmproj", VLM_MMPROJ,
         "-ngl", VLM_NGL, "--ctx-size", "1536", "--host", "0.0.0.0",
         "--port", "8080", "-np", "1", "--image-min-tokens", "64", "--image-max-tokens", "1024", "-b", "1024", "-ub", "512"],
        stdout=log_file, stderr=subprocess.STDOUT, env=env,
    )
    import requests
    t0 = time.time()
    for _ in range(120):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode}); see results/vlm_server.log")
        try:
            if requests.get("http://localhost:8080/health").status_code == 200:
                print(f"[VLM] Server ready after {time.time() - t0:.1f}s")
                return proc, round(time.time() - t0, 3)
        except requests.ConnectionError:
            pass
    raise RuntimeError("llama-server not ready after 120s")

def stop_vlm_server(proc):
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=10)
    print("[VLM] Server stopped")

def query_vlm(image_path, prompt=PROMPT, max_tokens=16):
    import requests
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = requests.post(VLM_URL, json={
        "model": "qwen2.5-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens, "temperature": 0,
    }, timeout=120)
    elapsed = time.time() - t0
    return resp.json()["choices"][0]["message"]["content"].strip(), round(elapsed, 3)

# ── Grounding: Qwen locates the jersey number, box scaled back to original resolution ──

BBOX_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")

def ground(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    s = (GROUND_TARGET_TOKENS * 28 * 28 / (w * h)) ** 0.5
    nw, nh = max(28, int(w * s / 28) * 28), max(28, int(h * s / 28) * 28)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    small_path = os.path.join(CROP_DIR, f"ground_{stem}.jpg")
    img.resize((nw, nh)).save(small_path, quality=95)

    raw, t = query_vlm(small_path, prompt=GROUND_PROMPT, max_tokens=64)
    m = BBOX_RE.search(raw)
    if not m:
        return None, raw, t
    x1, y1, x2, y2 = map(float, m.groups())
    if x2 > nw * 1.05 or y2 > nh * 1.05:
        return None, f"OUT_OF_RANGE({nw}x{nh}): {raw}", t
    sx, sy = w / nw, h / nh
    x1, x2 = sorted((x1 * sx, x2 * sx))
    y1, y2 = sorted((y1 * sy, y2 * sy))
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, raw, t
    return [x1, y1, x2, y2], raw, t

def box_crop(image_path, box):
    """Crop to the grounding box with generous padding (numbers sit on a torso; give context)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = box
    px = max(32, int((x2 - x1) * 0.5))
    py = max(32, int((y2 - y1) * 0.5))
    crop = img.crop((max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(CROP_DIR, f"crop_vbox_{stem}.jpg")
    crop.save(out, quality=95)
    return out, f"{crop.size[0]}x{crop.size[1]}"

# ── Scoring ──

def normalize(text):
    m = re.search(r"\d{1,2}", text)
    return m.group(0) if m else None

def score(pred_raw, truth):
    """Single-number truth: exact match. Multi-player truth ('6,30,32'): hit if prediction is any of them."""
    pred = normalize(pred_raw)
    targets = [t.strip() for t in truth.split(",")]
    return pred is not None and pred in targets, pred

def load_manifest():
    with open(MANIFEST) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    return data

def summarize(rows, key):
    valid = [r for r in rows if key in r]
    if not valid:
        return None
    correct = sum(r[key]["correct"] for r in valid)
    by_diff = {}
    for r in valid:
        d = by_diff.setdefault(r["difficulty"], [0, 0])
        d[0] += r[key]["correct"]
        d[1] += 1
    lat = [r[key]["vlm_time_s"] for r in valid]
    return {
        "accuracy": round(correct / len(rows), 3),   # over ALL images: a missing prediction counts as wrong
        "correct": correct, "answered": len(valid), "missing": len(rows) - len(valid),
        "by_difficulty": {k: f"{c}/{n}" for k, (c, n) in sorted(by_diff.items())},
        "vlm_latency_mean_s": round(statistics.mean(lat), 3),
        "vlm_latency_p50_s": round(statistics.median(lat), 3),
    }

READ_MAX_TOKENS = 256   # reading accuracy was best at <=256 image tokens (59% full-image run)

def prep_read(image_path):
    """Resize up OR down to ~READ_MAX_TOKENS 28px patches: equal token budget for every condition."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    s = (READ_MAX_TOKENS * 28 * 28 / (w * h)) ** 0.5
    nw, nh = max(28, int(w * s / 28) * 28), max(28, int(h * s / 28) * 28)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out = os.path.join(CROP_DIR, f"read_{stem}.jpg")
    img.resize((nw, nh), Image.LANCZOS).save(out, quality=95)
    return out

def run_condition(row, key, tag, image_path, truth, extra=None):
    try:
        raw, t = query_vlm(prep_read(image_path))
        ok, pred = score(raw, truth)
        row[key] = {"raw": raw, "pred": pred, "correct": ok, "vlm_time_s": t, **(extra or {})}
        print(f"[{tag}] {row['image']}: pred={pred} truth={truth} {'✓' if ok else '✗'} ({t}s)")
    except Exception as e:
        print(f"[{tag}] {row['image']}: ERROR {e}")

# ── Cloud tier (Meta Model API) ──

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def muse_condition(row, key, tag, img_or_path, truth, extra=None):
    from router import meta_client as mc
    try:
        raw, t = mc.muse_read(img_or_path)
        ok, pred = score(raw, truth)
        row[key] = {"raw": raw, "pred": pred, "correct": ok, "vlm_time_s": t, **(extra or {})}
        print(f"[{tag}] {row['image']}: pred={pred} truth={truth} {'✓' if ok else '✗'} ({t}s)")
    except Exception as e:
        print(f"[{tag}] {row['image']}: ERROR {e}")

def run_cloud_conditions(row, image_path, truth):
    from router import meta_client as mc
    name = row["image"]
    stem = os.path.splitext(name)[0]
    muse_condition(row, "muse_full", "MUSE", image_path, truth)
    try:
        boxes, raw, t_sam = mc.sam_segment(image_path)
    except Exception as e:
        print(f"[SAM3] {name}: ERROR {e}")
        return
    row["cloud_sam"] = {"boxes": boxes, "sam_time_s": t_sam}
    if not boxes:
        print(f"[SAM3] {name}: no 'jersey number' found ({t_sam}s)")
        return
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))   # largest = most legible instance
    print(f"[SAM3] {name}: {len(boxes)} box(es), using {box} ({t_sam}s)")
    img = Image.open(image_path).convert("RGB")
    crop = mc.pad_crop(img, box)
    crop_path = os.path.join(CROP_DIR, f"crop_cloud_{stem}.jpg")
    crop.save(crop_path, quality=95)
    extra = {"crop_size": f"{crop.size[0]}x{crop.size[1]}", "sam_time_s": t_sam, "n_boxes": len(boxes)}
    run_condition(row, "cloud_sam_edge_read", "C>EDGE", crop_path, truth, extra)
    muse_condition(row, "cloud_sam_muse", "C>MUSE", crop, truth, extra)

# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="also query full images without SAM")
    ap.add_argument("--grounding", action="store_true", help="Qwen box crop and Qwen box → SAM crop")
    ap.add_argument("--cloud", action="store_true", help="Meta API: Muse full, SAM 3.1 → edge read, SAM 3.1 → Muse")
    args = ap.parse_args()

    manifest = load_manifest()
    image_paths = [os.path.join(IMAGE_DIR, m["image"]) for m in manifest]

    # 0. Guard: llama-server needs fresh CMA (r36.4.7 leaks it on every process exit)
    cma_free = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("CmaFree"))
    if cma_free < 900_000:
        sys.exit(f"[CMA] Only {cma_free // 1024} MB free (need ~900+). Reboot first: sudo reboot")
    print(f"[CMA] {cma_free // 1024} MB free")

    # 1. VLM first — grab clean CMA before SAM fragments it
    vlm_proc, server_startup = start_vlm_server()
    rows, sam_load_time, boxes = [], None, {}
    try:
        # 2. Warm-up
        warm = os.path.join(RESULTS_DIR, "warmup.jpg")
        Image.open(image_paths[0]).convert("RGB").resize((224, 224)).save(warm)
        raw, t = query_vlm(warm)
        print(f"[VLM] Warm-up OK: '{raw}' in {t}s")

        # 3. Grounding pass (VLM only, before SAM touches the GPU)
        if args.grounding:
            for p in image_paths:
                name = os.path.basename(p)
                box, raw, t = ground(p)
                boxes[name] = {"box": box, "raw": raw, "ground_time_s": t}
                print(f"[GRND] {name}: {box if box else 'NO BOX'} ({t}s)" + ("" if box else f" raw={raw[:60]!r}"))

        # 4. SAM batch while server stays loaded
        jobs = [{"path": p, "box": boxes.get(os.path.basename(p), {}).get("box")} for p in image_paths]
        crops, sam_load_time = run_sam_batch(jobs)

        # 5. Query every condition
        for m in manifest:
            if vlm_proc.poll() is not None:
                print(f"[VLM] Server died (code {vlm_proc.returncode}) — aborting. See results/vlm_server.log")
                break
            name, truth = m["image"], m["jersey_number"]
            row = {"image": name, "truth": truth, "difficulty": m.get("difficulty", "?")}
            c = crops.get(name)
            if c:
                run_condition(row, "sam_crop", "CROP", c["crop_path"], truth,
                              {"crop_size": c["crop_size"], "sam_seg_time_s": c["sam_seg_time"]})
            if args.baseline:
                run_condition(row, "full_image", "FULL", os.path.join(IMAGE_DIR, name), truth)
            if args.grounding:
                g = boxes.get(name, {})
                row["grounding"] = g
                if g.get("box"):
                    vb_path, vb_size = box_crop(os.path.join(IMAGE_DIR, name), g["box"])
                    run_condition(row, "vlm_box", "VBOX", vb_path, truth,
                                  {"crop_size": vb_size, "ground_time_s": g["ground_time_s"]})
                if c and "box_crop_path" in c:
                    run_condition(row, "vlm_box_sam", "VSAM", c["box_crop_path"], truth,
                                  {"crop_size": c["box_crop_size"], "box_score": c["box_score"],
                                   "ground_time_s": g.get("ground_time_s")})
            if args.cloud:
                run_cloud_conditions(row, os.path.join(IMAGE_DIR, name), truth)
            rows.append(row)
    finally:
        stop_vlm_server(vlm_proc)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"order": "vlm_first", "sam_device": SAM_DEVICE, "vlm_ngl": VLM_NGL,
                   "image_tokens": "read=256 (equalized), ground~1000", "ubatch": 512, "ctx_size": 1536, "prompt": PROMPT,
                   "ground_prompt": GROUND_PROMPT if args.grounding else None,
                   "ground_target_tokens": GROUND_TARGET_TOKENS if args.grounding else None,
                   "cloud_models": "sam-3.1 + muse-spark-1.3" if args.cloud else None,
                   "sam_load_time_s": sam_load_time, "vlm_server_startup_s": server_startup},
        "sam_crop": summarize(rows, "sam_crop"),
        "full_image": summarize(rows, "full_image"),
        "vlm_box": summarize(rows, "vlm_box"),
        "vlm_box_sam": summarize(rows, "vlm_box_sam"),
        "muse_full": summarize(rows, "muse_full"),
        "cloud_sam_edge_read": summarize(rows, "cloud_sam_edge_read"),
        "cloud_sam_muse": summarize(rows, "cloud_sam_muse"),
    }
    out = os.path.join(RESULTS_DIR, f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(f"\n{'=' * 50}\nSUMMARY\n{json.dumps(summary, indent=2)}\n{'=' * 50}")
    print(f"[LOG] Wrote {out}")

if __name__ == "__main__":
    main()
