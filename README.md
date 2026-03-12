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

## Syncthing Setup (Phone → PC)
Use Syncthing to sync your phone gallery into one or more local folders, then point the watcher at those folders.

1. Install Syncthing on phone + desktop and pair devices.
2. Create a Syncthing folder on desktop (example): `~/Desktop/photobank`
3. Sync your phone gallery into that folder.
4. Start the watcher:

```bash
python watch_pipeline.py \
  --watch "/Users/allanodora/Desktop/photobank" \
  --output output
```

For multiple Syncthing folders, repeat `--watch`:

```bash
python watch_pipeline.py \
  --watch "/Users/allanodora/Desktop/Photos3" \
  --watch "/Users/allanodora/Desktop/Social media IMG" \
  --watch "/Users/allanodora/Desktop/photobank" \
  --output output
```

## Core Algorithm (Pipeline)
1. **Ingest**: Watch one or more sync folders for new files.
2. **Filter**:
   - Reject width < 800px
   - Reject blur score < 100 (variance of Laplacian)
3. **Deduplicate**: Perceptual hash (pHash) with Hamming distance threshold.
4. **AI Tagging**: CLIP tags into `people`, `events`, `stage/performance`, `artwork`, `buildings`, `logos`.
5. **AI Ranking**: Score = 0.4 * resolution + 0.4 * sharpness + 0.2 * brightness.
6. **Human Review**: Previews go to `output/review/`.
7. **Optimize**: WebP conversions into `hero/`, `sections/`, `gallery/` and categorized folders.
8. **Reporting**: `metadata/report.json` and `metadata/catalog.json`.

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
