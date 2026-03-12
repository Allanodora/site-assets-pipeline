#!/usr/bin/env python3
"""
Local GUI to browse/search/select curated media and copy to a chosen folder.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request, send_file


def load_catalog(catalog_path: Path) -> List[Dict[str, str]]:
    if not catalog_path.exists():
        return []
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return data


def safe_path(base_dir: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base_dir.resolve())
        return True
    except Exception:
        return False


def create_app(base_dir: Path, catalog_path: Path) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_file(Path(__file__).parent / "templates" / "index.html")

    @app.get("/api/items")
    def items():
        q = request.args.get("q", "").strip().lower()
        category = request.args.get("category", "").strip().lower()
        sort = request.args.get("sort", "latest")
        limit = int(request.args.get("limit", "200"))

        catalog = load_catalog(catalog_path)
        results: List[Dict[str, str]] = []

        for item in catalog:
            src = item.get("src", "")
            cat = item.get("category", "").lower()
            if category and category != cat:
                continue
            if q and q not in src.lower() and q not in cat:
                continue
            results.append(item)

        def sort_key(it: Dict[str, str]):
            src = Path(it.get("src", ""))
            try:
                return src.stat().st_mtime
            except Exception:
                return 0

        reverse = sort == "latest"
        results.sort(key=sort_key, reverse=reverse)

        return jsonify({"items": results[:limit], "total": len(results)})

    @app.get("/api/image")
    def image():
        src = request.args.get("src", "")
        if not src:
            return ("missing src", 400)
        path = Path(src)
        if not path.exists():
            return ("not found", 404)
        return send_file(path)

    @app.post("/api/copy")
    def copy_selected():
        payload = request.get_json(force=True)
        dest = Path(payload.get("dest", "")).expanduser()
        items = payload.get("items", [])
        if not dest:
            return jsonify({"ok": False, "error": "missing destination"}), 400
        dest.mkdir(parents=True, exist_ok=True)

        copied = 0
        for item in items:
            src = Path(item.get("src", ""))
            if not src.exists():
                continue
            # Copy the optimized file if it exists, otherwise source.
            target = item.get("target", "")
            optimized = None
            if target:
                # Try to locate optimized webp in output folders.
                stem = src.stem + ".webp"
                candidate = base_dir / target / stem
                if candidate.exists() and safe_path(base_dir, candidate):
                    optimized = candidate
            copy_src = optimized if optimized else src
            shutil.copy2(copy_src, dest / copy_src.name)
            copied += 1

        return jsonify({"ok": True, "copied": copied})

    @app.get("/api/pick_dir")
    def pick_dir():
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory()
            root.destroy()
            if not folder:
                return jsonify({"ok": False, "error": "no folder selected"}), 200
            return jsonify({"ok": True, "path": folder})
        except Exception:
            # Fallback to macOS AppleScript chooser if tkinter isn't available.
            try:
                out = subprocess.check_output(
                    ["osascript", "-e", "POSIX path of (choose folder)"],
                    stderr=subprocess.STDOUT,
                    text=True,
                ).strip()
                if not out:
                    return jsonify({"ok": False, "error": "no folder selected"}), 200
                return jsonify({"ok": True, "path": out})
            except Exception as exc:
                return jsonify(
                    {"ok": False, "error": f"folder picker failed: {exc}"}
                ), 500

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local GUI for curated media.")
    parser.add_argument("--base", default="output", help="Output base folder")
    parser.add_argument("--catalog", default="metadata/catalog.json", help="Catalog JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser().resolve()
    catalog_path = Path(args.catalog).expanduser().resolve()

    app = create_app(base_dir, catalog_path)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
