"""
Download a small WLASL subset and lay it out for train.py.

Uses the reduced WLASL release on Hugging Face (jherng/wlasl_reduced: 394 clips,
35 glosses, with signer bounding boxes), keeps the N glosses with the most clips,
crops every clip to a square around the signer, and writes:

    <out>/videos/<gloss>/<id>.mp4
    <out>/train.txt  <out>/val.txt      "<gloss>/<id>.mp4 <class_id>"
    <out>/class_names.json              {"<class_id>": "<gloss>"}

Usage:
    python scripts/prepare_wlasl_subset.py --out dataset --glosses 10 --val-per-class 3

Requires ffmpeg on PATH. WLASL is released for academic, non-commercial use.
"""

import argparse
import collections
import concurrent.futures as cf
import csv
import io
import json
import os
import random
import subprocess
import urllib.parse
import urllib.request

BASE = "https://huggingface.co/datasets/jherng/wlasl_reduced/resolve/main/"


def fetch(path: str) -> bytes:
    req = urllib.request.Request(BASE + urllib.parse.quote(path), headers={"User-Agent": "asl-plugin"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def crop_clip(row: dict, src: str, dst: str, size: int) -> None:
    """Square crop around the signer's bounding box, resized to size x size."""
    W, H = int(row["video_width"]), int(row["video_height"])
    x, y, w, h = (float(row[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    s = min(max(w * W, h * H), W, H)
    x0 = int(max(0, min(W - s, cx - s / 2)))
    y0 = int(max(0, min(H - s, cy - s / 2)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"crop={int(s)}:{int(s)}:{x0}:{y0},scale={size}:{size}", "-an", dst],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--glosses", type=int, default=10, help="keep the N glosses with the most clips")
    ap.add_argument("--val-per-class", type=int, default=3)
    ap.add_argument("--size", type=int, default=160, help="side of the cropped square, in pixels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = list(csv.DictReader(io.StringIO(fetch("metadata.csv").decode())))
    gloss_ids = json.loads(fetch("gloss_map.json"))

    by_gloss = collections.defaultdict(list)
    for r in rows:
        by_gloss[r["gloss"]].append(r)
    chosen = sorted(by_gloss, key=lambda g: (-len(by_gloss[g]), g))[: args.glosses]
    print(f"Glosses: {', '.join(chosen)}")

    raw_dir = os.path.join(args.out, "_raw")
    jobs = [r for g in chosen for r in by_gloss[g]]

    def prepare(r):
        rel = r["filepath"].split("videos/", 1)[1]
        raw = os.path.join(raw_dir, rel)
        dst = os.path.join(args.out, "videos", rel)
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            with open(raw, "wb") as f:
                f.write(fetch(r["filepath"]))
            crop_clip(r, raw, dst, args.size)
        return rel

    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(prepare, jobs))

    random.seed(args.seed)
    train, val = [], []
    for g in chosen:
        clips = sorted(r["filepath"].split("videos/", 1)[1] for r in by_gloss[g])
        random.shuffle(clips)
        cid = gloss_ids[g]
        val += [f"{c} {cid}" for c in clips[: args.val_per_class]]
        train += [f"{c} {cid}" for c in clips[args.val_per_class:]]

    for name, lines in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(args.out, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump({str(gloss_ids[g]): g for g in chosen}, f, indent=2)
    print(f"{len(train)} train / {len(val)} val clips written to {args.out}/")


if __name__ == "__main__":
    main()
