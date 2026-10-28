"""
Benchmark harness: all images → SAM (one process) → crops → VLM (one server session) → score.
Usage:
  python benchmark/run_benchmark.py              # SAM-crop condition
  python benchmark/run_benchmark.py --baseline   # also query full images (no SAM) for ablation
Reboot before each full run so CMA starts clean (r36.4.7 leaks CMA per process exit).
"""
import sys, os, re, json, time, base64, subprocess, statistics, argparse
from datetime import datetime

# === CONFIG ===
IMAGE_DIR = "data/images"
MANIFEST = "data/manifest.json"
CROP_DIR = "results/crops"
RESULTS_DIR = "results"
LLAMA_SERVER = "/mnt/nvme/project/llama.cpp/build/bin/llama-server"
VLM_MODEL = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
VLM_MMPROJ = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf"
VLM_URL = "http://localhost:8080/v1/chat/completions"
SAM_DEVICE = "cuda"
VLM_NGL = "10"
PROMPT = "What jersey number is the player wearing? Reply with ONLY the number, nothing else."

os.makedirs(CROP_DIR, exist_ok=True)

# ── Stage 1: SAM batch worker (loads model ONCE, segments every image) ──

SAM_BATCH_SCRIPT = r'''
import sys, time, json, os, numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

image_paths = json.loads(sys.argv[1])
crop_dir, device = sys.argv[2], sys.argv[3]

t0 = time.time()
model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml",
                   "/mnt/nvme/models/sam2/sam2.1_hiera_tiny.pt", device=device)
predictor = SAM2ImagePredictor(model)
print(json.dumps({"event": "loaded", "sam_load_time": round(time.time() - t0, 3)}), flush=True)

for path in image_paths:
    try:
        image_np = np.array(Image.open(path).convert("RGB"))
        h, w = image_np.shape[:2]
        t1 = time.time()
        predictor.set_image(image_np)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, h // 2]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        seg_time = time.time() - t1
        mask = masks[int(np.argmax(scores))]
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        py, px = int((y1 - y0) * 0.1), int((x1 - x0) * 0.1)
        y0, y1 = max(0, y0 - py), min(h, y1 + py)
        x0, x1 = max(0, x0 - px), min(w, x1 + px)
        crop = image_np[y0:y1, x0:x1]
        name = os.path.splitext(os.path.basename(path))[0]
        crop_path = os.path.join(crop_dir, f"crop_{name}.jpg")
        Image.fromarray(crop).save(crop_path)
        print(json.dumps({
            "event": "crop", "image": os.path.basename(path), "crop_path": crop_path,
            "crop_size": f"{crop.shape[1]}x{crop.shape[0]}", "image_size": f"{w}x{h}",
            "sam_seg_time": round(seg_time, 3),
            "sam_scores": [round(float(s), 4) for s in scores],
        }), flush=True)
    except Exception as e:
        print(json.dumps({"event": "error", "image": os.path.basename(path), "error": str(e)}), flush=True)
'''

def run_sam_batch(image_paths):
    print(f"[SAM] Batch segmenting {len(image_paths)} images (device={SAM_DEVICE})...")
    proc = subprocess.run(
        [sys.executable, "-c", SAM_BATCH_SCRIPT, json.dumps(image_paths), CROP_DIR, SAM_DEVICE],
        capture_output=True, text=True,
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
            print(f"[SAM] {ev['image']}: {ev['crop_size']} in {ev['sam_seg_time']}s")
        elif ev["event"] == "error":
            print(f"[SAM] {ev['image']}: ERROR {ev['error']}")
    if proc.returncode != 0:
        print(f"[SAM] Process exited with code {proc.returncode}:\n{proc.stderr[-2000:]}")
    return crops, load_time

# ── Stage 2: one llama-server session for all queries ──

def start_vlm_server():
    print(f"[VLM] Starting llama-server (ngl={VLM_NGL}, unified memory)...")
    log_file = open(os.path.join(RESULTS_DIR, "vlm_server.log"), "w")
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", VLM_MODEL, "--mmproj", VLM_MMPROJ,
         "-ngl", VLM_NGL, "--ctx-size", "512", "--host", "0.0.0.0",
         "--port", "8080", "-np", "1", "--image-max-tokens", "256"],
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

def query_vlm(image_path):
    import requests
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = requests.post(VLM_URL, json={
        "model": "qwen2.5-vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": 16, "temperature": 0,
    }, timeout=120)
    elapsed = time.time() - t0
    return resp.json()["choices"][0]["message"]["content"].strip(), round(elapsed, 3)

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
    if isinstance(data, dict):  # tolerate {"images": [...]} style
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
        "accuracy": round(correct / len(valid), 3), "correct": correct, "total": len(valid),
        "by_difficulty": {k: f"{c}/{n}" for k, (c, n) in sorted(by_diff.items())},
        "vlm_latency_mean_s": round(statistics.mean(lat), 3),
        "vlm_latency_p50_s": round(statistics.median(lat), 3),
    }

# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="also query full images without SAM")
    args = ap.parse_args()

    manifest = load_manifest()
    image_paths = [os.path.join(IMAGE_DIR, m["image"]) for m in manifest]

    # 1. VLM FIRST — grab clean CMA before SAM fragments it
    vlm_proc, server_startup = start_vlm_server()
    rows = []
    try:
        # 2. Warm-up: confirm image inference works before spending time on SAM
        from PIL import Image
        warm = os.path.join(RESULTS_DIR, "warmup.jpg")
        Image.open(image_paths[0]).convert("RGB").resize((224, 224)).save(warm)
        raw, t = query_vlm(warm)
        print(f"[VLM] Warm-up OK: '{raw}' in {t}s")

        # 3. SAM batch while server stays loaded
        crops, sam_load_time = run_sam_batch(image_paths)

        # 4. Query all crops
        for m in manifest:
            if vlm_proc.poll() is not None:
                print(f"[VLM] Server died (code {vlm_proc.returncode}) — aborting. See results/vlm_server.log")
                break
            row = {"image": m["image"], "truth": m["jersey_number"], "difficulty": m.get("difficulty", "?")}
            c = crops.get(m["image"])
            if c:
                try:
                    raw, t = query_vlm(c["crop_path"])
                    ok, pred = score(raw, m["jersey_number"])
                    row["sam_crop"] = {"raw": raw, "pred": pred, "correct": ok, "vlm_time_s": t,
                                       "sam_seg_time_s": c["sam_seg_time"], "crop_size": c["crop_size"]}
                    print(f"[CROP] {m['image']}: pred={pred} truth={m['jersey_number']} {'✓' if ok else '✗'} ({t}s)")
                except Exception as e:
                    print(f"[CROP] {m['image']}: ERROR {e}")
            if args.baseline:
                try:
                    raw, t = query_vlm(os.path.join(IMAGE_DIR, m["image"]))
                    ok, pred = score(raw, m["jersey_number"])
                    row["full_image"] = {"raw": raw, "pred": pred, "correct": ok, "vlm_time_s": t}
                    print(f"[FULL] {m['image']}: pred={pred} truth={m['jersey_number']} {'✓' if ok else '✗'} ({t}s)")
                except Exception as e:
                    print(f"[FULL] {m['image']}: ERROR {e}")
            rows.append(row)
    finally:
        stop_vlm_server(vlm_proc)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {"order": "vlm_first", "sam_device": SAM_DEVICE, "vlm_ngl": VLM_NGL, "prompt": PROMPT,
                   "sam_load_time_s": locals().get("sam_load_time"), "vlm_server_startup_s": server_startup},
        "sam_crop": summarize(rows, "sam_crop"),
        "full_image": summarize(rows, "full_image"),
    }
    out = os.path.join(RESULTS_DIR, f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print(f"\n{'=' * 50}\nSUMMARY\n{json.dumps(summary, indent=2)}\n{'=' * 50}")
    print(f"[LOG] Wrote {out}")

if __name__ == "__main__":
    main()
