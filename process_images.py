#!/usr/bin/env python3
"""
Process a folder of images into optimized, categorized, website-ready assets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORIES = ["people", "events", "stage/performance", "artwork", "buildings", "logos"]
CATEGORY_PROMPTS = {
    "people": "people",
    "events": "an event",
    "stage/performance": "a stage performance",
    "artwork": "artwork",
    "buildings": "a building",
    "logos": "a logo",
}
DEFAULT_DUP_THRESHOLD = 5  # Hamming distance for perceptual hash


@dataclass
class ImageInfo:
    src_path: str
    width: int
    height: int
    phash: str
    blur_score: float
    brightness: float


@dataclass
class TaggedImage:
    src_path: str
    category: str
    score: float
    width: int
    height: int
    blur_score: float
    brightness: float
    outputs: Dict[str, str]


# -----------------------------
# Utilities
# -----------------------------


def iter_images(input_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for root, _, files in os.walk(input_dir):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in SUPPORTED_EXTS:
                paths.append(Path(root) / name)
    return paths


def ensure_dirs(base_output: Path) -> Dict[str, Path]:
    dirs = {
        "output": base_output,
        "hero": base_output / "hero",
        "sections": base_output / "sections",
        "gallery": base_output / "gallery",
        "categorized": base_output / "categorized",
        "rejected_duplicates": base_output / "../rejected/duplicates",
        "rejected_blurry": base_output / "../rejected/blurry",
        "rejected_small": base_output / "../rejected/small",
        "metadata": base_output / "../metadata",
    }

    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES + ["unclassified"]:
        (dirs["categorized"] / sanitize_category(category)).mkdir(parents=True, exist_ok=True)

    return dirs


def sanitize_category(category: str) -> str:
    return category.replace("/", "_").replace(" ", "_")


def compute_blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def analyze_image(path: str) -> ImageInfo | None:
    try:
        img = Image.open(path).convert("RGB")
        width, height = img.size
        phash = str(imagehash.phash(img))
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        blur_score = compute_blur_score(gray)
        brightness = compute_brightness(gray)
        return ImageInfo(
            src_path=path,
            width=width,
            height=height,
            phash=phash,
            blur_score=blur_score,
            brightness=brightness,
        )
    except Exception:
        return None


def phash_distance(h1: str, h2: str) -> int:
    return imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)


def is_duplicate(phash: str, known_hashes: List[str], threshold: int) -> bool:
    for h in known_hashes:
        if phash_distance(phash, h) <= threshold:
            return True
    return False


def load_clip_model():
    import torch
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


def tag_with_clip(
    model,
    preprocess,
    tokenizer,
    device: str,
    image_paths: List[str],
    categories: List[str],
) -> Dict[str, str]:
    import torch

    texts = [f"a photo of {CATEGORY_PROMPTS.get(c, c)}" for c in categories]
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    results: Dict[str, str] = {}

    for path in tqdm(image_paths, desc="Tagging with CLIP"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                idx = int(similarity[0].argmax().item())
                results[path] = categories[idx]
        except Exception:
            results[path] = "unclassified"

    return results


def normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v - min_v == 0:
        return [0.5 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def score_images(infos: List[ImageInfo]) -> Dict[str, float]:
    resolutions = [i.width * i.height for i in infos]
    sharpness = [i.blur_score for i in infos]
    brightness = [i.brightness for i in infos]

    res_n = normalize_scores(resolutions)
    sharp_n = normalize_scores(sharpness)
    bright_n = normalize_scores(brightness)

    scores: Dict[str, float] = {}
    for info, r, s, b in zip(infos, res_n, sharp_n, bright_n):
        # Weighted score: resolution and sharpness matter slightly more than brightness
        scores[info.src_path] = 0.4 * r + 0.4 * s + 0.2 * b
    return scores


def resize_and_save(src_path: str, dest_path: Path, width: int) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path).convert("RGB") as img:
        w, h = img.size
        if w <= width:
            resized = img
        else:
            new_h = int(h * (width / w))
            resized = img.resize((width, new_h), Image.LANCZOS)
        resized.save(dest_path, format="WEBP", quality=80, method=6)


def copy_to_rejected(src_path: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(src_path).name
    shutil.copy2(src_path, dest)


def analyze_image_path(path: Path) -> ImageInfo | None:
    return analyze_image(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate and optimize website images.")
    parser.add_argument("--input", required=True, help="Input folder with raw images")
    parser.add_argument("--output", required=True, help="Output base folder")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--max-hero-images", type=int, default=6)
    parser.add_argument("--max-gallery-images", type=int, default=0)

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    dirs = ensure_dirs(output_dir)

    image_paths = iter_images(input_dir)
    if not image_paths:
        print("No images found.")
        return

    # Analyze images (hash, blur, size, brightness) with multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    infos: List[ImageInfo] = []
    with ProcessPoolExecutor() as ex:
        for info in tqdm(ex.map(analyze_image_path, image_paths), total=len(image_paths), desc="Analyzing"):
            if info is not None:
                infos.append(info)

    duplicates_removed = 0
    blurry_removed = 0
    small_removed = 0

    unique_infos: List[ImageInfo] = []
    known_hashes: List[str] = []

    # Duplicate detection and quality filtering
    for info in tqdm(infos, desc="Filtering"):
        if info.width < 800:
            copy_to_rejected(info.src_path, dirs["rejected_small"])
            small_removed += 1
            continue

        if info.blur_score < args.blur_threshold:
            copy_to_rejected(info.src_path, dirs["rejected_blurry"])
            blurry_removed += 1
            continue

        if is_duplicate(info.phash, known_hashes, DEFAULT_DUP_THRESHOLD):
            copy_to_rejected(info.src_path, dirs["rejected_duplicates"])
            duplicates_removed += 1
            continue

        known_hashes.append(info.phash)
        unique_infos.append(info)

    if not unique_infos:
        print("No images left after filtering.")
        return

    # Tagging
    print("Loading CLIP model...")
    model, preprocess, tokenizer, device = load_clip_model()
    tags = tag_with_clip(
        model, preprocess, tokenizer, device, [i.src_path for i in unique_infos], CATEGORIES
    )

    # Scoring
    scores = score_images(unique_infos)

    # Group by category
    categorized: Dict[str, List[ImageInfo]] = {c: [] for c in CATEGORIES}
    uncategorized: List[ImageInfo] = []
    for info in unique_infos:
        cat = tags.get(info.src_path, "unclassified")
        if cat in categorized:
            categorized[cat].append(info)
        else:
            uncategorized.append(info)

    # Sort each category by score
    for cat in categorized:
        categorized[cat].sort(key=lambda i: scores[i.src_path], reverse=True)

    # Hero selection: top 1 per category, then fill by best overall
    hero_candidates: List[ImageInfo] = []
    for cat in CATEGORIES:
        if categorized[cat]:
            hero_candidates.append(categorized[cat][0])

    # Add remaining best images overall
    remaining = [i for cat in categorized.values() for i in cat if i not in hero_candidates]
    remaining.sort(key=lambda i: scores[i.src_path], reverse=True)
    hero_candidates.extend(remaining)
    hero_images = hero_candidates[: max(args.max_hero_images, 0)]

    # Section selection: top 3 per category excluding heroes
    hero_set = {i.src_path for i in hero_images}
    section_images: List[ImageInfo] = []
    for cat in CATEGORIES:
        items = [i for i in categorized[cat] if i.src_path not in hero_set]
        section_images.extend(items[:3])

    # Gallery selection: remaining, optionally capped
    section_set = {i.src_path for i in section_images}
    gallery_images = [
        i for cat in categorized.values() for i in cat if i.src_path not in hero_set and i.src_path not in section_set
    ]
    gallery_images.sort(key=lambda i: scores[i.src_path], reverse=True)

    if args.max_gallery_images and args.max_gallery_images > 0:
        gallery_images = gallery_images[: args.max_gallery_images]

    # Optimization outputs
    tagged_images: List[TaggedImage] = []

    for info in tqdm(unique_infos, desc="Optimizing"):
        cat = tags.get(info.src_path, "unclassified")
        cat_dir = dirs["categorized"] / sanitize_category(cat)
        cat_dir.mkdir(parents=True, exist_ok=True)
        out_name = Path(info.src_path).stem + ".webp"
        categorized_path = cat_dir / out_name
        resize_and_save(info.src_path, categorized_path, 1200)

        outputs: Dict[str, str] = {"categorized": str(categorized_path)}

        if info.src_path in hero_set:
            hero_path = dirs["hero"] / out_name
            resize_and_save(info.src_path, hero_path, 1920)
            outputs["hero"] = str(hero_path)
        elif info.src_path in section_set:
            section_path = dirs["sections"] / out_name
            resize_and_save(info.src_path, section_path, 1200)
            outputs["sections"] = str(section_path)
        else:
            gallery_path = dirs["gallery"] / out_name
            resize_and_save(info.src_path, gallery_path, 800)
            outputs["gallery"] = str(gallery_path)

        tagged_images.append(
            TaggedImage(
                src_path=info.src_path,
                category=cat,
                score=scores[info.src_path],
                width=info.width,
                height=info.height,
                blur_score=info.blur_score,
                brightness=info.brightness,
                outputs=outputs,
            )
        )

    # Metadata output
    tags_path = dirs["metadata"] / "image_tags.json"
    report_path = dirs["metadata"] / "summary_report.json"

    with open(tags_path, "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tagged_images], f, indent=2)

    category_counts = {cat: len(categorized[cat]) for cat in CATEGORIES}
    if uncategorized:
        category_counts["unclassified"] = len(uncategorized)
    report = {
        "total_images_processed": len(image_paths),
        "duplicates_removed": duplicates_removed,
        "blurry_images_removed": blurry_removed,
        "small_images_removed": small_removed,
        "final_optimized_images": len(tagged_images),
        "category_counts": category_counts,
        "hero_images": len(hero_images),
        "section_images": len(section_images),
        "gallery_images": len(gallery_images),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(f"Metadata: {tags_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
