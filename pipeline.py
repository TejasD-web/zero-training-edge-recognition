"""
End-to-end dry pass: image → SAM 2.1 → crop → VLM → jersey number prediction
Usage: python pipeline.py <image_path>

Runs SAM in a subprocess so GPU memory is fully released before VLM loads.
"""
import sys
import time
import json
import os
import subprocess

# === CONFIG ===
VLM_URL = "http://localhost:8080/v1/chat/completions"
CROP_DIR = "results/crops"
LLAMA_SERVER = "/mnt/nvme/project/llama.cpp/build/bin/llama-server"
VLM_MODEL = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
VLM_MMPROJ = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf"

os.makedirs(CROP_DIR, exist_ok=True)

# ── SAM stage (runs as a separate script) ──

SAM_SCRIPT = '''
import sys, time, json, numpy as np, torch, os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM_CHECKPOINT = "/mnt/nvme/models/sam2/sam2.1_hiera_tiny.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

image_path = sys.argv[1]
crop_dir = sys.argv[2]

img = Image.open(image_path).convert("RGB")
image_np = np.array(img)

t0 = time.time()
model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device="cpu")
predictor = SAM2ImagePredictor(model)
load_time = time.time() - t0

t1 = time.time()
predictor.set_image(image_np)
h, w = image_np.shape[:2]
masks, scores, logits = predictor.predict(
    point_coords=np.array([[w//2, h//2]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
seg_time = time.time() - t1

best_idx = np.argmax(scores)
mask = masks[best_idx]
rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)
y_min, y_max = np.where(rows)[0][[0, -1]]
x_min, x_max = np.where(cols)[0][[0, -1]]
pad_y = int((y_max - y_min) * 0.1)
pad_x = int((x_max - x_min) * 0.1)
y_min = max(0, y_min - pad_y)
y_max = min(h, y_max + pad_y)
x_min = max(0, x_min - pad_x)
x_max = min(w, x_max + pad_x)
crop_np = image_np[y_min:y_max, x_min:x_max]

crop_path = os.path.join(crop_dir, "crop_best.jpg")
Image.fromarray(crop_np).save(crop_path)

result = {
    "crop_path": crop_path,
    "crop_size": f"{crop_np.shape[1]}x{crop_np.shape[0]}",
    "sam_load_time": round(load_time, 3),
    "sam_seg_time": round(seg_time, 3),
    "scores": [round(float(s), 4) for s in scores],
    "image_size": f"{w}x{h}",
}
print(json.dumps(result))
'''

# ── VLM helpers ──

def start_vlm_server():
    print("[VLM] Starting llama-server...")
    log_file = open("results/vlm_server.log", "w")
    clean_env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    proc = subprocess.Popen(
        [LLAMA_SERVER,
         "-m", VLM_MODEL,
         "--mmproj", VLM_MMPROJ,
         "-ngl", "0",
         "--ctx-size", "512",
         "--host", "0.0.0.0",
         "--port", "8080",
         "-np", "1"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
            env=clean_env,
    )
    import requests
    for i in range(120):
        time.sleep(1)
        try:
            r = requests.get("http://localhost:8080/health")
            if r.status_code == 200:
                print(f"[VLM] Server ready after {i+1}s")
                return proc
        except requests.ConnectionError:
            pass
    print("[VLM] WARNING: server may not be ready after 120s")
    return proc

def stop_vlm_server(proc):
    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)
    print("[VLM] Server stopped")

def query_vlm(crop_path):
    import requests, base64
    with open(crop_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    resp = requests.post(VLM_URL, json={
        "model": "qwen2.5-vl",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "What jersey number is the player wearing? Reply with ONLY the number, nothing else."}
            ]}
        ],
        "max_tokens": 16,
        "temperature": 0
    })
    elapsed = time.time() - t0
    result = resp.json()
    answer = result["choices"][0]["message"]["content"].strip()
    print(f"[VLM] Predicted: '{answer}' in {elapsed:.2f}s")
    return answer, elapsed

# ── Main ──

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"{'='*50}\n")

    # Stage 1: Run SAM in a SEPARATE PROCESS so GPU memory is fully freed on exit
    print("[SAM] Running segmentation in subprocess...")
    sam_result = subprocess.run(
        [sys.executable, "-c", SAM_SCRIPT, image_path, CROP_DIR],
        capture_output=True, text=True,
        cwd=os.getcwd(),
    )
    if sam_result.returncode != 0:
        print(f"[SAM] FAILED:\n{sam_result.stderr}")
        sys.exit(1)

    sam_info = json.loads(sam_result.stdout.strip().split('\n')[-1])
    print(f"[SAM] Loaded in {sam_info['sam_load_time']}s")
    print(f"[SAM] Segmented in {sam_info['sam_seg_time']}s, scores: {sam_info['scores']}")
    print(f"[CROP] Saved {sam_info['crop_size']} to {sam_info['crop_path']}")
    print(f"[MEM] SAM process exited — GPU memory fully released")
    time.sleep(5)  # let CMA recover

    sam_time = sam_info['sam_load_time'] + sam_info['sam_seg_time']
    crop_path = sam_info['crop_path']

    # Stage 2: Start VLM on clean GPU, run inference, stop VLM
    vlm_proc = start_vlm_server()
    try:
        prediction, vlm_time = query_vlm(crop_path)
    except Exception as e:
        print(f"[VLM] Error: {e}")
        prediction, vlm_time = "ERROR", 0
    finally:
        stop_vlm_server(vlm_proc)

    # Log result
    result = {
        "image": image_path,
        "prediction": prediction,
        "sam_time_s": round(sam_time, 3),
        "vlm_time_s": round(vlm_time, 3),
        "total_time_s": round(sam_time + vlm_time, 3),
        "crop_size": sam_info['crop_size'],
        "image_size": sam_info['image_size'],
    }

    print(f"\n{'='*50}")
    print(f"RESULT: {json.dumps(result, indent=2)}")
    print(f"{'='*50}")

    log_path = "results/runs.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[LOG] Appended to {log_path}")

if __name__ == "__main__":
    main()
