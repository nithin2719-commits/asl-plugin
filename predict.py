"""
Inference for a trained checkpoint.

    python predict.py --checkpoint checkpoints/best_model.pth --video clip.mp4 \
        --class_names dataset/class_names.json

Frames are sampled and normalised exactly as in dataset.WLASLDataset, using the
num_frames / frame_size stored in the checkpoint.
"""

import argparse
import json
import os
import pickle
import time
from typing import Optional

import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import create_model


def read_frames(video_path: str, num_frames: int) -> np.ndarray:
    """Uniformly sample num_frames RGB frames, padding short clips with the last frame."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        return vr.get_batch(_indices(len(vr), num_frames)).asnumpy()  # (T, H, W, C) uint8
    except ImportError:
        import torchvision.io as video_io
        video, _, _ = video_io.read_video(video_path, pts_unit="sec")
        return video[_indices(video.shape[0], num_frames)].numpy()


def _indices(total: int, num_frames: int) -> np.ndarray:
    if total >= num_frames:
        return np.linspace(0, total - 1, num_frames, dtype=np.int64)
    return np.concatenate([np.arange(total), np.full(num_frames - total, total - 1, dtype=np.int64)])


class SignPredictor:
    def __init__(self, checkpoint: str, label_encoder: Optional[str] = None,
                 class_names: Optional[str] = None, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.meta = {k: v for k, v in ckpt.items() if not k.endswith("state_dict")}
        self.num_frames = int(ckpt.get("num_frames", 32))
        size = ckpt.get("frame_size", 224)
        self.frame_size = (size, size) if isinstance(size, int) else tuple(size)

        self.model = create_model(num_classes=ckpt["num_classes"], backbone=ckpt.get("backbone", "r3d_18"),
                                  pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        encoder_path = label_encoder or os.path.join(os.path.dirname(checkpoint), "label_encoder.pkl")
        with open(encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        names = {}
        if class_names and os.path.exists(class_names):
            with open(class_names) as f:
                names = json.load(f)
        self.labels = [names.get(str(c), str(c)) for c in self.label_encoder.classes_]

        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        x = torch.nn.functional.interpolate(x, size=self.frame_size, mode="bilinear", align_corners=False)
        x = (x / 255.0).permute(1, 0, 2, 3)  # (C, T, H, W)
        return ((x - self.mean) / self.std).unsqueeze(0)

    @torch.no_grad()
    def predict(self, video_path: str, top_k: int = 5) -> dict:
        t0 = time.perf_counter()
        frames = read_frames(video_path, self.num_frames)
        t1 = time.perf_counter()
        probs = torch.softmax(self.model(self.preprocess(frames).to(self.device)), dim=1)[0].cpu()
        t2 = time.perf_counter()
        conf, idx = probs.topk(min(top_k, len(self.labels)))
        return {
            "label": self.labels[idx[0]],
            "top": [{"label": self.labels[i], "confidence": float(c)} for c, i in zip(conf, idx)],
            "frames": frames,
            "timing_ms": {"decode": round((t1 - t0) * 1000, 1), "model": round((t2 - t1) * 1000, 1)},
        }


def main():
    ap = argparse.ArgumentParser(description="Predict the sign in a video clip")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--label_encoder", default=None)
    ap.add_argument("--class_names", default=None, help="JSON mapping class id -> gloss")
    ap.add_argument("--video", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    p = SignPredictor(args.checkpoint, args.label_encoder, args.class_names)
    out = p.predict(args.video, args.top_k)
    print(f"{os.path.basename(args.video)} -> {out['label']}")
    for t in out["top"]:
        print(f"  {t['label']:<16} {t['confidence']:6.1%}")


if __name__ == "__main__":
    main()
