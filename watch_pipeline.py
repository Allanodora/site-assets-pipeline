#!/usr/bin/env python3
"""
Watch a phone-synced folder and process new images/videos into website-ready assets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov"}
SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS

CATEGORIES = ["people", "events", "stage/performance", "artwork", "buildings", "logos"]
CATEGORY_PROMPTS = {
    "people": "people",
    "events": "an event",
    "stage/performance": "a stage performance",
    "artwork": "artwork",
    "buildings": "a building",
    "logos": "a logo",
}

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None


@dataclass
class Report:
    total_processed: int = 0
    rejected: int = 0
    optimized: int = 0
    videos_processed: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_processed": self.total_processed,
            "rejected": self.rejected,
            "optimized": self.optimized,
            "videos_processed": self.videos_processed,
        }


def setup_dirs(base_output: Path) -> Dict[str, Path]:
    dirs = {
        "output": base_output,
        "hero": base_output / "hero",
        "sections": base_output / "sections",
        "gallery": base_output / "gallery",
        "videos": base_output / "videos",
        "thumbnails": base_output / "thumbnails",
        "review": base_output / "review",
        "categorized": base_output / "categorized",
        "rejected_blurry": base_output / "../rejected/blurry",
        "rejected_small": base_output / "../rejected/small",
        "rejected_duplicates": base_output / "../rejected/duplicates",
        "logs": base_output / "../logs",
        "metadata": base_output / "../metadata",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES + ["unclassified"]:
        (dirs["categorized"] / sanitize_category(category)).mkdir(parents=True, exist_ok=True)
    return dirs


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("watch_pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def is_stable_file(path: Path, wait_seconds: float = 1.5) -> bool:
    if not path.exists():
        return False
    size1 = path.stat().st_size
    time.sleep(wait_seconds)
    if not path.exists():
        return False
    size2 = path.stat().st_size
    return size1 == size2 and size1 > 0


def compute_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash_distance(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def sanitize_category(category: str) -> str:
    return category.replace("/", "_").replace(" ", "_")


def load_clip_model() -> Tuple[object, object, object, str]:
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE

    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    _CLIP_MODEL = model
    _CLIP_PREPROCESS = preprocess
    _CLIP_TOKENIZER = tokenizer
    _CLIP_DEVICE = device
    return model, preprocess, tokenizer, device


def tag_with_clip(image_path: Path) -> str:
    import torch

    model, preprocess, tokenizer, device = load_clip_model()
    texts = [f"a photo of {CATEGORY_PROMPTS.get(c, c)}" for c in CATEGORIES]
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        idx = int(similarity[0].argmax().item())
        return CATEGORIES[idx]


def score_image(width: int, height: int, blur_score: float, brightness: float) -> float:
    # Normalize to rough ranges for single-image scoring.
    resolution_score = min(1.0, (width * height) / (3000 * 2000))
    sharpness_score = min(1.0, blur_score / 500.0)
    brightness_score = 1.0 - min(1.0, abs(brightness - 127.0) / 127.0)
    return 0.4 * resolution_score + 0.4 * sharpness_score + 0.2 * brightness_score


def resize_and_save(src_path: Path, dest_path: Path, width: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path).convert("RGB") as img:
        w, h = img.size
        if w <= width:
            resized = img
        else:
            new_h = int(h * (width / w))
            resized = img.resize((width, new_h), Image.LANCZOS)
        resized.save(dest_path, format="WEBP", quality=80, method=6)


def copy_rejected(src_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_dir / src_path.name)


def process_image(
    path_str: str,
    output_dir: str,
    blur_threshold: float,
    dup_threshold: int,
    shared_hashes,
    shared_lock,
    hero_threshold: float,
    section_threshold: float,
) -> Dict[str, str]:
    src = Path(path_str)
    try:
        img = Image.open(src).convert("RGB")
        width, height = img.size
        if width < 800:
            copy_rejected(src, Path(output_dir) / "../rejected/small")
            return {"status": "rejected", "reason": "small"}

        phash = str(imagehash.phash(img))
        with shared_lock:
            for h in list(shared_hashes):
                if phash_distance(phash, h) <= dup_threshold:
                    copy_rejected(src, Path(output_dir) / "../rejected/duplicates")
                    return {"status": "rejected", "reason": "duplicate"}
            shared_hashes.append(phash)

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        blur_score = compute_blur_score(gray)
        if blur_score < blur_threshold:
            copy_rejected(src, Path(output_dir) / "../rejected/blurry")
            return {"status": "rejected", "reason": "blurry"}

        brightness = float(np.mean(gray))
        category = tag_with_clip(src)
        score = score_image(width, height, blur_score, brightness)

        out_base = Path(output_dir)
        stem = src.stem + ".webp"

        # Always write a review preview.
        resize_and_save(src, out_base / "review" / stem, 1200)

        # Categorized folder
        cat_dir = out_base / "categorized" / sanitize_category(category)
        resize_and_save(src, cat_dir / stem, 1200)

        # Route to website folders by score thresholds.
        if score >= hero_threshold:
            resize_and_save(src, out_base / "hero" / stem, 1920)
            target = "hero"
        elif score >= section_threshold:
            resize_and_save(src, out_base / "sections" / stem, 1200)
            target = "sections"
        else:
            resize_and_save(src, out_base / "gallery" / stem, 800)
            target = "gallery"

        return {
            "status": "optimized",
            "reason": "image",
            "category": category,
            "score": f"{score:.4f}",
            "target": target,
            "width": str(width),
            "height": str(height),
            "blur_score": f"{blur_score:.2f}",
            "brightness": f"{brightness:.2f}",
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def process_video(path_str: str, output_dir: str) -> Dict[str, str]:
    src = Path(path_str)
    out_base = Path(output_dir)
    out_video = out_base / "videos" / f"{src.stem}.mp4"
    out_thumb = out_base / "thumbnails" / f"{src.stem}.jpg"

    cmd_video = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        "scale=-2:1080",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(out_video),
    ]

    cmd_thumb = [
        "ffmpeg",
        "-y",
        "-ss",
        "00:00:01",
        "-i",
        str(src),
        "-frames:v",
        "1",
        str(out_thumb),
    ]

    try:
        subprocess.run(cmd_video, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(cmd_thumb, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"status": "optimized", "reason": "video"}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


class MediaHandler(FileSystemEventHandler):
    def __init__(self, queue: Queue, logger: logging.Logger):
        self.queue = queue
        self.logger = logger

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if is_hidden_path(path):
            return
        if path.suffix.lower() in SUPPORTED_EXTS:
            self.logger.info(f"Detected new file: {path}")
            self.queue.put(path)


def enqueue_existing(watch_dir: Path, queue: Queue) -> List[Path]:
    files: List[Path] = []
    for root, _, names in os.walk(watch_dir):
        for name in names:
            p = Path(root) / name
            if is_hidden_path(p):
                continue
            if p.suffix.lower() in SUPPORTED_EXTS:
                files.append(p)
    for p in files:
        queue.put(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a folder and process new media.")
    parser.add_argument(
        "--watch",
        action="append",
        required=True,
        help="Folder to watch (repeatable for multiple folders)",
    )
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--dup-threshold", type=int, default=5)
    parser.add_argument("--hero-threshold", type=float, default=0.80)
    parser.add_argument("--section-threshold", type=float, default=0.60)
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() or 2, 2))
    args = parser.parse_args()

    watch_dirs = [Path(p) for p in args.watch]
    output_dir = Path(args.output)
    dirs = setup_dirs(output_dir)
    logger = setup_logger(dirs["logs"] / "pipeline.log")

    report_path = dirs["metadata"] / "report.json"
    report = Report()
    report_lock = threading.Lock()
    catalog_lock = threading.Lock()
    catalog_path = dirs["metadata"] / "catalog.json"
    catalog: List[Dict[str, str]] = []
    if catalog_path.exists():
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog = json.load(f)
        except Exception:
            catalog = []

    from multiprocessing import Manager

    manager = Manager()
    shared_hashes = manager.list()
    shared_lock = manager.Lock()

    queue: Queue = Queue()

    def handle_result(path: Path, result: Dict[str, str]) -> None:
        with report_lock:
            report.total_processed += 1
            if result.get("status") == "optimized":
                report.optimized += 1
                if result.get("reason") == "video":
                    report.videos_processed += 1
                if result.get("reason") == "image":
                    logger.info(
                        f"Optimized image ({result.get('category')}, score {result.get('score')}, {result.get('target')}): {path}"
                    )
                else:
                    logger.info(f"Optimized video: {path}")
            elif result.get("status") == "rejected":
                report.rejected += 1
                logger.info(f"Rejected ({result.get('reason')}): {path}")
            else:
                report.rejected += 1
                logger.error(f"Error processing {path}: {result.get('reason')}")

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

        if result.get("status") == "optimized" and result.get("reason") == "image":
            with catalog_lock:
                catalog.append(
                    {
                        "src": str(path),
                        "category": result.get("category", "unclassified"),
                        "score": result.get("score", "0"),
                        "target": result.get("target", "gallery"),
                        "width": result.get("width", "0"),
                        "height": result.get("height", "0"),
                        "blur_score": result.get("blur_score", "0"),
                        "brightness": result.get("brightness", "0"),
                    }
                )
                with open(catalog_path, "w", encoding="utf-8") as f:
                    json.dump(catalog, f, indent=2)

    def consumer_loop():
        ex: ProcessPoolExecutor | None = None
        inflight: Dict[object, Path] = {}
        try:
            ex = ProcessPoolExecutor(max_workers=args.workers)
            while True:
                while not queue.empty() and len(inflight) < args.workers * 2:
                    path = queue.get()
                    if path is None:
                        queue.task_done()
                        return
                    if not is_stable_file(path):
                        logger.info(f"Skipped unstable file: {path}")
                        queue.task_done()
                        continue
                    ext = path.suffix.lower()
                    try:
                        if ext in IMAGE_EXTS:
                            fut = ex.submit(
                                process_image,
                                str(path),
                                str(output_dir),
                                args.blur_threshold,
                                args.dup_threshold,
                                shared_hashes,
                                shared_lock,
                                args.hero_threshold,
                                args.section_threshold,
                            )
                        else:
                            fut = ex.submit(process_video, str(path), str(output_dir))
                        inflight[fut] = path
                    except Exception as exc:
                        logger.error(f"Process pool error, retrying: {exc}")
                        # Recreate pool if it broke.
                        if ex is not None:
                            ex.shutdown(wait=False, cancel_futures=True)
                        ex = ProcessPoolExecutor(max_workers=args.workers)
                        queue.put(path)
                    finally:
                        queue.task_done()

                if not inflight:
                    time.sleep(0.2)
                    continue

                try:
                    for fut in as_completed(list(inflight.keys()), timeout=0.5):
                        path = inflight.pop(fut)
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = {"status": "error", "reason": str(exc)}
                        handle_result(path, result)
                        break
                except TimeoutError:
                    pass
        finally:
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)

    consumer = threading.Thread(target=consumer_loop, daemon=True)
    consumer.start()

    existing: List[Path] = []
    for wd in watch_dirs:
        existing.extend(enqueue_existing(wd, queue))
    if existing:
        for _ in tqdm(range(len(existing)), desc="Queued existing media"):
            time.sleep(0.01)

    observer = Observer()
    handler = MediaHandler(queue, logger)
    for wd in watch_dirs:
        observer.schedule(handler, str(wd), recursive=True)
    observer.start()

    logger.info("Watching folders for new media:")
    for wd in watch_dirs:
        logger.info(f" - {wd}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        observer.stop()
        observer.join()
        queue.put(None)
        consumer.join()


if __name__ == "__main__":
    main()
