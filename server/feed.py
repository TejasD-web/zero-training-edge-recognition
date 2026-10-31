"""
Feed frames to the router service, like a camera would.

  python server/feed.py clip.mp4 --fps 2              # video -> frames (ffmpeg) -> /read
  python server/feed.py data/own/dev/                 # every jpg/png in a folder
  python server/feed.py a.jpg b.jpg --manifest data/own/dev/manifest.json   # attach truth for scoring

Frames are sent one after another (the server serializes them anyway); --delay adds a pause
between frames so the dashboard is watchable on video.
"""
import os, sys, json, glob, time, argparse, subprocess, tempfile
import requests

IMG_EXT = (".jpg", ".jpeg", ".png")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def video_frames(path, fps, out_dir):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-vf", f"fps={fps}",
                    "-q:v", "2", os.path.join(out_dir, "frame_%05d.jpg")], check=True)
    return sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))


def load_truth(manifest):
    if not manifest:
        return {}
    data = json.load(open(manifest))
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    return {m["image"]: str(m.get("truth", m.get("jersey_number"))) for m in data}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="video file(s), image file(s) or folder(s)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second sampled from video")
    ap.add_argument("--delay", type=float, default=0.0, help="pause between frames (s)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--manifest", help="JSON list of {image, truth|jersey_number} to score answers")
    args = ap.parse_args()

    truth = load_truth(args.manifest)
    tmp = tempfile.TemporaryDirectory()
    frames = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            frames += sorted(p for p in glob.glob(os.path.join(inp, "*")) if p.lower().endswith(IMG_EXT))
        elif inp.lower().endswith(VID_EXT):
            d = tempfile.mkdtemp(dir=tmp.name)
            frames += video_frames(inp, args.fps, d)
        elif inp.lower().endswith(IMG_EXT):
            frames.append(inp)
        else:
            sys.exit(f"don't know what to do with {inp}")
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        sys.exit("no frames found")

    print(f"[FEED] {len(frames)} frames -> {args.url}/read")
    for i, p in enumerate(frames, 1):
        data = {}
        t = truth.get(os.path.basename(p))
        if t:
            data["truth"] = t
        with open(p, "rb") as f:
            r = requests.post(f"{args.url}/read", files={"image": (os.path.basename(p), f, "image/jpeg")},
                              data=data, timeout=180)
        if r.status_code != 200:
            print(f"[FEED] {p}: HTTP {r.status_code} {r.text[:200]}")
            continue
        rec = r.json()
        conf = (rec.get("edge_conf") or {}).get("joint")
        where = "QUEUED" if rec.get("queued") else ("CLOUD" if rec["left_device"] else "EDGE")
        mark = "" if rec.get("correct") is None else ("  ✓" if rec["correct"] else "  ✗")
        print(f"[{i:>3}/{len(frames)}] {os.path.basename(p):<20} {str(rec['answer']):>6}  "
              f"conf {conf if conf is not None else '-':<6} {where:<6} {rec['total_s']}s{mark}")
        if args.delay:
            time.sleep(args.delay)
    tmp.cleanup()


if __name__ == "__main__":
    main()
