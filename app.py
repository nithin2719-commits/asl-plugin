"""
Web UI for a trained checkpoint: play a clip, see what the model predicts, the
frames it actually looked at, the training curves and the validation confusion matrix.

    python app.py --checkpoint checkpoints/best_model.pth \
        --root_dir dataset --val_file dataset/val.txt --class_names dataset/class_names.json

Then open http://localhost:8000. Any clip can also be uploaded from the page.
"""

import argparse
import base64
import json
import os
import tempfile

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from predict import SignPredictor
from utils import count_parameters

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_UPLOAD = 64 * 1024 * 1024


def thumbnails(frames: np.ndarray, count: int = 8, size: int = 160) -> list:
    """Evenly pick `count` of the sampled frames and return them as JPEG data URLs."""
    picks = np.linspace(0, len(frames) - 1, count).astype(int)
    out = []
    for i in picks:
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        scale = size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append({"index": int(i), "src": "data:image/jpeg;base64," + base64.b64encode(buf).decode()})
    return out


def build_app(args) -> FastAPI:
    predictor = SignPredictor(args.checkpoint, args.label_encoder, args.class_names)
    video_dir = os.path.join(args.root_dir, "videos") if args.root_dir else None
    names = {}
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            names = json.load(f)

    clips = []
    if args.val_file and os.path.exists(args.val_file):
        with open(args.val_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    clips.append({"path": parts[0], "label": names.get(parts[1], parts[1])})

    history_path = os.path.join(os.path.dirname(args.checkpoint), "history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    evaluation = {}

    def clip_file(rel: str) -> str:
        if not video_dir:
            raise HTTPException(404, "no --root_dir configured")
        full = os.path.realpath(os.path.join(video_dir, rel))
        if not full.startswith(os.path.realpath(video_dir) + os.sep) or not os.path.isfile(full):
            raise HTTPException(404, "clip not found")
        return full

    def run(path: str) -> dict:
        res = predictor.predict(path, top_k=5)
        res["frames"] = thumbnails(res["frames"])
        res["num_frames"] = predictor.num_frames
        return res

    app = FastAPI(title="ASL sign classifier", version="1.0")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/info")
    def info():
        m = predictor.meta
        return {
            "backbone": m.get("backbone"),
            "labels": predictor.labels,
            "num_frames": predictor.num_frames,
            "frame_size": list(predictor.frame_size),
            "best_val_acc": m.get("best_val_acc"),
            "epoch": m.get("epoch"),
            "parameters": count_parameters(predictor.model),
            "device": str(predictor.device),
            "history": history,
            "clips": clips,
        }

    @app.get("/clips/{rel:path}")
    def clip(rel: str):
        return FileResponse(clip_file(rel), media_type="video/mp4")

    @app.post("/api/predict")
    async def predict(request: Request, clip: str = ""):
        if clip:
            return run(clip_file(clip))
        body = await request.body()
        if not body:
            raise HTTPException(400, "send a video file as the request body, or ?clip=<path>")
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "video too large")
        suffix = os.path.splitext(request.headers.get("x-filename", "clip.mp4"))[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(body)
        try:
            return run(tmp.name)
        finally:
            os.unlink(tmp.name)

    @app.get("/api/evaluate")
    def evaluate():
        """Accuracy and confusion matrix over --val_file (computed once, then cached)."""
        if not clips:
            raise HTTPException(404, "no --val_file configured")
        if not evaluation:
            labels = predictor.labels
            index = {l: i for i, l in enumerate(labels)}
            matrix = [[0] * len(labels) for _ in labels]
            rows = []
            for c in clips:
                r = predictor.predict(clip_file(c["path"]), top_k=1)
                matrix[index[c["label"]]][index[r["label"]]] += 1
                rows.append({"path": c["path"], "label": c["label"], "predicted": r["label"],
                             "confidence": r["top"][0]["confidence"]})
            correct = sum(r["label"] == r["predicted"] for r in rows)
            evaluation.update({"labels": labels, "matrix": matrix, "rows": rows,
                               "accuracy": correct / len(rows), "correct": correct, "total": len(rows)})
        return evaluation

    return app


def main():
    ap = argparse.ArgumentParser(description="Web UI for a trained ASL video classifier")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--label_encoder", default=None, help="defaults to label_encoder.pkl next to the checkpoint")
    ap.add_argument("--class_names", default=None, help="JSON mapping class id -> gloss")
    ap.add_argument("--root_dir", default=None, help="dataset root containing videos/")
    ap.add_argument("--val_file", default=None, help="annotation file listed as sample clips")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
