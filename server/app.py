"""
FastAPI service + live dashboard around the confidence router.

The server owns llama-server for its whole lifetime (started on boot, stopped on shutdown),
so every request is a warm ~1 s edge read. Requests are serialized (llama-server runs -np 1).

Run (from the repo root, fresh-ish memory):
  uvicorn server.app:app --host 0.0.0.0 --port 8000
Then open http://<jetson-ip>:8000 and feed frames:
  python server/feed.py clip.mp4 --fps 2
  python server/feed.py data/own/dev/            # a folder of images

Endpoints:
  POST /read          image upload (+ optional truth) -> routed answer, confidence, tier
  GET  /events        recent routed frames (dashboard polls this)
  GET  /mode          {"offline": bool}
  POST /mode          {"offline": bool}  -- "cut the link" for the demo
  GET  /queue         offline queue entries
  POST /sync          verify queued frames with Muse Spark
  POST /reset         clear events + queue (before a demo take)
"""
import os, sys, json, shutil, threading, contextlib, io
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from router import route as rt              # chdirs to ROOT; brings rb (benchmark) + mc (Meta client)
rb = rt.rb

UPLOAD_DIR = os.path.join(rb.RESULTS_DIR, "uploads")    # gitignore this
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_EVENTS = 500

state = {"offline": False, "events": [], "next_id": 1}
route_lock = threading.Lock()     # one request at a time into llama-server
state_lock = threading.Lock()
server_proc = None


@contextlib.asynccontextmanager
async def lifespan(app):
    global server_proc
    mem_avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    if mem_avail < 4_500_000:
        sys.exit(f"[MEM] Only {mem_avail // 1024} MB available (need ~4.5 GB). "
                 "Drop caches first: sync && sudo sysctl -w vm.drop_caches=3")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    server_proc, _ = rb.start_vlm_server()
    try:
        yield
    finally:
        rb.stop_vlm_server(server_proc)


app = FastAPI(title="Edge confidence router", lifespan=lifespan)
app.mount("/frames", StaticFiles(directory=UPLOAD_DIR, check_dir=False), name="frames")


def _summary(events):
    n = len(events)
    if not n:
        return {"frames": 0}
    scored = [e for e in events if e.get("correct") is not None]
    lat = [e["total_s"] for e in events]
    return {
        "frames": n,
        "on_device": sum(not e["left_device"] and not e.get("queued") for e in events),
        "muse_calls": sum(e["muse_calls"] for e in events) + sum(1 for e in events if e.get("synced")),
        "queued": sum(bool(e.get("queued")) for e in events),
        "mean_latency_s": round(sum(lat) / n, 2),
        "scored": len(scored),
        "correct": sum(e["correct"] for e in scored),
    }


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/read")
def read(image: UploadFile = File(...), truth: str = Form(None)):
    # Sync def: FastAPI runs it in a worker thread, so the blocking edge/Muse calls don't stall the loop.
    if server_proc is not None and server_proc.poll() is not None:
        raise HTTPException(503, "llama-server died; see results/vlm_server.log and restart the service")
    ext = os.path.splitext(image.filename or "")[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(415, "jpg or png only")
    name = f"{datetime.now():%Y%m%d_%H%M%S_%f}{ext}"       # unique: also the queue's file name
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(image.file.read())

    max_tier = 1 if state["offline"] else 3
    with route_lock:
        rec = rt.route_confidence(path, max_tier=max_tier)
    rec["policy"] = "confidence"
    if rec.get("needs_queue"):
        rt.enqueue(path, rec, truth)
    if truth:
        rec["truth"] = truth
        rec["correct"] = rb.score(rec["answer"] or "", truth)[0]
    rec["source"] = image.filename
    rec["frame_url"] = f"/frames/{name}"
    rec["offline"] = state["offline"]
    rec["threshold"] = rt.CONF_THRESHOLD
    rec["at"] = datetime.now().isoformat(timespec="seconds")

    with state_lock:
        rec["id"] = state["next_id"]
        state["next_id"] += 1
        state["events"].append(rec)
        del state["events"][:-MAX_EVENTS]
    return rec


@app.get("/events")
def events(since: int = 0):
    with state_lock:
        evs = [e for e in state["events"] if e["id"] > since]
        summ = _summary(state["events"])
    return {"events": evs, "summary": summ, "offline": state["offline"], "threshold": rt.CONF_THRESHOLD}


class Mode(BaseModel):
    offline: bool


@app.get("/mode")
def get_mode():
    return {"offline": state["offline"]}


@app.post("/mode")
def set_mode(m: Mode):
    state["offline"] = m.offline
    return {"offline": state["offline"]}


def _queue_entries():
    if not os.path.exists(rt.QUEUE_FILE):
        return []
    return [json.loads(l) for l in open(rt.QUEUE_FILE) if l.strip()]


@app.get("/queue")
def queue():
    entries = _queue_entries()
    return {"pending": sum(e["status"] == "unverified" for e in entries), "entries": entries}


@app.post("/sync")
def sync():
    if state["offline"]:
        raise HTTPException(409, "offline mode is on; switch it off to sync")
    before = {e["image"]: e["status"] for e in _queue_entries()}
    pending = sum(s == "unverified" for s in before.values())
    if not pending:
        return {"verified": 0, "pending": 0, "log": "Queue is empty."}
    if not rt.cloud_available():
        raise HTTPException(503, f"Meta Model API unreachable; {pending} frames stay queued")
    buf = io.StringIO()
    with route_lock, contextlib.redirect_stdout(buf):
        try:
            rt.sync()
        except SystemExit as e:          # sync() exits if the link drops mid-way
            print(e)
    after = _queue_entries()
    newly = [e for e in after if before.get(e["image"]) == "unverified" and e["status"] == "verified"]
    # Reflect sync results on the dashboard events.
    by_img = {e["image"]: e for e in newly}
    with state_lock:
        for ev in state["events"]:
            q = by_img.get(ev["image"])
            if q:
                ev["synced"] = {"cloud_answer": q["cloud_answer"], "changed": q["corrected"],
                                "correct": q.get("correct")}
    return {
        "verified": len(newly),
        "pending": sum(e["status"] == "unverified" for e in after),
        "confirmed": sum(not e["corrected"] for e in newly),
        "changed": sum(e["corrected"] for e in newly),
        "fixed": sum(1 for e in newly if e.get("edge_correct") is False and e.get("correct")),
        "uploaded_mb": round(sum(e["bytes"] for e in newly) / 1e6, 2),
        "results": [{"image": e["image"], "edge": e["edge_answer"], "cloud": e["cloud_answer"],
                     "changed": e["corrected"]} for e in newly],
        "log": buf.getvalue(),
    }


@app.post("/reset")
def reset():
    with state_lock:
        state["events"].clear()
    shutil.rmtree(rt.QUEUE_DIR, ignore_errors=True)
    return {"ok": True}
