"""
End-to-end dry pass: image → SAM 2.1 → crop → VLM → jersey number prediction
Usage: python pipeline.py <image_path>
"""
import sys
import time
import json
import numpy as np
import torch
from PIL import Image
import subprocess
import signal
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# === CONFIG ===
SAM_CHECKPOINT = "/mnt/nvme/models/sam2/sam2.1_hiera_tiny.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
VLM_URL = "http://localhost:8080/v1/chat/completions"
CROP_DIR = "/mnt/nvme/project/hackathon/results/crops"
LLAMA_SERVER = "/mnt/nvme/project/llama.cpp/build/bin/llama-server"
VLM_MODEL = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-q4_k_m.gguf"
VLM_MMPROJ = "/mnt/nvme/models/qwen2.5-vl-3b-gguf/Qwen2.5-VL-3B-Instruct-mmproj-q8_0.gguf"

import os
os.makedirs(CROP_DIR, exist_ok=True)

def start_vlm_server():
    """Launch llama-server and wait until it's ready."""
    print("[VLM] Starting llama-server...")
    proc = subprocess.Popen(
        [LLAMA_SERVER,
         "-m", VLM_MODEL,
         "--mmproj", VLM_MMPROJ,
         "-ngl", "99",
         "--ctx-size", "2048",
         "--host", "0.0.0.0",
         "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Wait for server to be ready
    import requests
    for i in range(60):  # up to 60s
        time.sleep(1)
        try:
            r = requests.get("http://localhost:8080/health")
            if r.status_code == 200:
                print(f"[VLM] Server ready after {i+1}s")
                return proc
        except requests.ConnectionError:
            pass
    print("[VLM] WARNING: server may not be ready")
    return proc

def stop_vlm_server(proc):
    """Kill llama-server and wait for GPU memory to free."""
    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(2)  # let GPU memory fully release
    print("[VLM] Server stopped")

def load_sam():
    t0 = time.time()
    model = build_sam2(SAM_CONFIG, SAM_CHECKPOINT, device="cuda")
    predictor = SAM2ImagePredictor(model)
    print(f"[SAM] Loaded in {time.time()-t0:.2f}s")
    return predictor

def segment_players(predictor, image_np):
    """Use automatic mask generation or center-point prompt."""
    t0 = time.time()
    predictor.set_image(image_np)
    
    # For now: grid of points across the image to find players
    # TODO: replace with automatic mask generation or smarter prompting
    h, w = image_np.shape[:2]
    # Single center point as placeholder
    masks, scores, logits = predictor.predict(
        point_coords=np.array([[w//2, h//2]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    
    elapsed = time.time() - t0
    print(f"[SAM] Segmented in {elapsed:.2f}s, {len(masks)} masks, scores: {scores}")
    return masks, scores, elapsed

def mask_to_crop(image_np, mask):
    """Extract bounding box crop from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add 10% padding
    h, w = image_np.shape[:2]
    pad_y = int((y_max - y_min) * 0.1)
    pad_x = int((x_max - x_min) * 0.1)
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    
    return image_np[y_min:y_max, x_min:x_max]

def query_vlm(crop_path):
    """Send crop to local llama-server for jersey number reading."""
    import requests
    import base64
    
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"\n{'='*50}")
    print(f"Processing: {image_path}")
    print(f"{'='*50}\n")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    image_np = np.array(img)
    print(f"[INPUT] Image size: {image_np.shape[1]}x{image_np.shape[0]}")
    
    # Stage 1: SAM segmentation
    predictor = load_sam()
    masks, scores, sam_time = segment_players(predictor, image_np)
    
    # Use best mask
    best_idx = np.argmax(scores)
    crop_np = mask_to_crop(image_np, masks[best_idx])
    
    # Save crop
    crop_path = os.path.join(CROP_DIR, "crop_best.jpg")
    Image.fromarray(crop_np).save(crop_path)
    print(f"[CROP] Saved {crop_np.shape[1]}x{crop_np.shape[0]} to {crop_path}")
    
    # Unload SAM to free GPU memory
    del predictor
    del masks, logits
    torch.cuda.empty_cache()
    print(f"[MEM] SAM unloaded, GPU cache cleared")
    
    # Stage 2: Start VLM, run inference, stop VLM
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
        "crop_size": f"{crop_np.shape[1]}x{crop_np.shape[0]}",
    }
    
    print(f"\n{'='*50}")
    print(f"RESULT: {json.dumps(result, indent=2)}")
    print(f"{'='*50}")
    
    # Append to results log
    log_path = "results/runs.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[LOG] Appended to {log_path}")

if __name__ == "__main__":
    main()
