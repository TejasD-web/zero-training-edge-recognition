"""
Meta Model API client: SAM 3.1 concept segmentation + Muse Spark reading.
Test:  python router/meta_client.py img_001.jpg img_011.jpg img_014.jpg [--read]
"""
import os, re, sys, json, time, base64, io
import requests
from PIL import Image, ImageDraw

API_BASE = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
MUSE_MODEL = "muse-spark-1.3"
UPLOAD_LONG_SIDE = 1280   # downscale before upload; boxes are scaled back to original pixels

BOX_RE = re.compile(r"<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)\|>")

def load_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    with open(env_path) as f:
        for line in f:
            if line.strip().startswith("MODEL_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("MODEL_API_KEY not set (env var or .env)")

def _headers():
    return {"Authorization": f"Bearer {load_key()}", "Content-Type": "application/json"}

def _encode(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def _response_text(data):
    """Pull output_text out of a Responses API payload."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and "text" in c:
                parts.append(c["text"])
    return "".join(parts)

def sam_segment(image_path, prompt="jersey number", timeout=60):
    """Returns (boxes in ORIGINAL pixel coords, raw output_text, latency_s)."""
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    s = min(1.0, UPLOAD_LONG_SIDE / max(W, H))
    up = img.resize((int(W * s), int(H * s)), Image.LANCZOS) if s < 1 else img
    body = {
        "model": SAM_MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _encode(up)},
        ]}],
    }
    t0 = time.time()
    r = requests.post(f"{API_BASE}/responses", headers=_headers(), json=body, timeout=timeout)
    latency = round(time.time() - t0, 3)
    if r.status_code != 200:
        raise RuntimeError(f"SAM HTTP {r.status_code}: {r.text[:500]}")
    raw = _response_text(r.json())
    boxes = []
    for x1, y1, x2, y2, w, h in BOX_RE.findall(raw):
        sx, sy = W / int(w), H / int(h)   # w/h = image size the box is relative to
        boxes.append([int(int(x1) * sx), int(int(y1) * sy), int((int(x2) + 1) * sx), int((int(y2) + 1) * sy)])
    return boxes, raw, latency

def muse_read(image_path_or_img, prompt="What jersey number is shown? Reply with ONLY the number, nothing else.",
              timeout=90):
    """Returns (answer_text, latency_s)."""
    img = image_path_or_img if isinstance(image_path_or_img, Image.Image) else Image.open(image_path_or_img).convert("RGB")
    body = {
        "model": MUSE_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _encode(img)}},
        ]}],
        "max_completion_tokens": 2048,      # reasoning tokens count toward this
        "reasoning_effort": "low",
    }
    t0 = time.time()
    r = requests.post(f"{API_BASE}/chat/completions", headers=_headers(), json=body, timeout=timeout)
    latency = round(time.time() - t0, 3)
    if r.status_code != 200:
        raise RuntimeError(f"Muse HTTP {r.status_code}: {r.text[:500]}")
    return r.json()["choices"][0]["message"]["content"].strip(), latency

def pad_crop(img, box, frac=0.5, min_pad=24):
    W, H = img.size
    x1, y1, x2, y2 = box
    px, py = max(min_pad, int((x2 - x1) * frac)), max(min_pad, int((y2 - y1) * frac))
    return img.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_read = "--read" in sys.argv
    out_dir = "results/cloud_viz"
    os.makedirs(out_dir, exist_ok=True)
    for name in args or ["img_001.jpg", "img_011.jpg", "img_014.jpg"]:
        path = os.path.join("data/images", name)
        try:
            boxes, raw, t = sam_segment(path)
        except Exception as e:
            print(f"[SAM3.1] {name}: ERROR {e}")
            continue
        print(f"[SAM3.1] {name}: {len(boxes)} box(es) in {t}s -> {boxes}")
        if not boxes:
            print(f"         raw: {raw[:300]!r}")
        img = Image.open(path).convert("RGB")
        viz = img.copy()
        d = ImageDraw.Draw(viz)
        lw = max(4, img.width // 150)
        for i, b in enumerate(boxes):
            d.rectangle(b, outline=(0, 255, 0), width=lw)
            crop = pad_crop(img, b)
            stem = os.path.splitext(name)[0]
            crop.save(os.path.join(out_dir, f"{stem}_box{i}.jpg"))
            if do_read:
                try:
                    ans, rt = muse_read(crop)
                    print(f"         box{i} -> Muse Spark: {ans!r} ({rt}s)")
                except Exception as e:
                    print(f"         box{i} -> Muse Spark ERROR {e}")
        viz.thumbnail((800, 800))
        viz.save(os.path.join(out_dir, name))
    print(f"[LOG] overlays + crops in {out_dir}/")
