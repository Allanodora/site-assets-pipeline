# Media Curation CLI

A command-line tool that curates large image collections into optimized, categorized, website-ready assets, including a folder-watching pipeline for phone-synced media.

## Features
- Accepts JPG, PNG, WebP input folders
- Perceptual-hash duplicate removal
- Blur detection (variance of Laplacian)
- Minimum width filtering (800px)
- CLIP-based content tagging into:
  - people
  - events
  - stage/performance
  - artwork
  - buildings
  - logos
- Best-image scoring (resolution, sharpness, brightness)
- Website-ready WebP outputs (hero/sections/gallery)
- JSON metadata and summary report
- Multiprocessing + tqdm progress bars
- Folder watching with watchdog for phone-synced media
- Video optimization with ffmpeg (H.264 + thumbnail)

## Output Structure
```
project/
    process_images.py
    watch_pipeline.py
    input_images/
    output/
        hero/
        sections/
        gallery/
        videos/
        thumbnails/
        review/
        categorized/
            people/
            events/
            artwork/
    rejected/
        duplicates/
        blurry/
        small/
    metadata/
        image_tags.json
        summary_report.json
        report.json
        catalog.json
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python process_images.py --input input_images --output output
```

Optional arguments:
- `--blur-threshold` (default `100`)
- `--max-hero-images` (default `6`)
- `--max-gallery-images` (default `0`, meaning unlimited)

## Example
```bash
python process_images.py \
  --input input_images \
  --output output \
  --blur-threshold 120 \
  --max-hero-images 8 \
  --max-gallery-images 200
```

## Watch Mode (Phone Sync)
Point the watcher at the local folder your tethering/sync app writes to.

```bash
python watch_pipeline.py --watch phone_media --output output
```

Multiple folders:
```bash
python watch_pipeline.py \
  --watch "/Users/allanodora/Desktop/Photos3" \
  --watch "/Users/allanodora/Desktop/Social media IMG" \
  --watch "/Users/allanodora/Desktop/photobank" \
  --output output
```

Optional arguments:
- `--blur-threshold` (default `100`)
- `--dup-threshold` (default `5`)
- `--hero-threshold` (default `0.80`)
- `--section-threshold` (default `0.60`)
- `--workers` (default: CPU count)

The watcher creates a `output/review/` preview for quick human review and writes
metadata to `metadata/catalog.json` as images are processed.

## GUI Picker (Fastest to Ship)
Run a local browser app to search, filter, and copy selected images into any folder.

```bash
python gallery_app.py --base output --catalog metadata/catalog.json
```

Then open `http://127.0.0.1:5050` in your browser.

## Notes
- CLIP model is downloaded on first run by `open-clip-torch`.
- If no GPU is available, tagging runs on CPU.
- `ffmpeg` must be installed for video processing.
