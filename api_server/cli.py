#!/usr/bin/env python3
"""CLI for the fashion-search API. Designed for easy use from Claude Code.

Usage:
    python api_server/cli.py text "red maroon t-shirt" --k 5
    python api_server/cli.py image /path/to/photo.jpg --k 3
    python api_server/cli.py image /path/to/photo.jpg --k 3 --save-dir /tmp/results
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

DEFAULT_BASE = "http://localhost:8080"


def search_text(base_url: str, query: str, k: int):
    resp = httpx.get(f"{base_url}/search/text", params={"q": query, "k": k}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_image(base_url: str, image_path: str, k: int):
    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{base_url}/search/image", params={"k": k}, files={"file": f}, timeout=30
        )
    resp.raise_for_status()
    return resp.json()


def download_image(base_url: str, image_url: str, dest: Path):
    resp = httpx.get(f"{base_url}{image_url}", timeout=15)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def print_results(results, base_url: str, save_dir: Path | None):
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['productDisplayName']}")
        print(f"   {r['baseColour']} {r['articleType']} | {r['gender']} | {r['season']}")
        print(f"   {base_url}{r['image_url']}")
        if save_dir:
            dest = save_dir / f"{r['id']}.jpg"
            download_image(base_url, r["image_url"], dest)
            print(f"   saved: {dest}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Fashion catalog search")
    parser.add_argument("--url", default=DEFAULT_BASE, help="API base URL")
    sub = parser.add_subparsers(dest="command", required=True)

    text_p = sub.add_parser("text", help="Search by text description")
    text_p.add_argument("query", help="Text query")
    text_p.add_argument("--k", type=int, default=5)
    text_p.add_argument("--save-dir", type=Path, help="Download result images to this dir")
    text_p.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")

    img_p = sub.add_parser("image", help="Search by image file")
    img_p.add_argument("path", help="Path to image file")
    img_p.add_argument("--k", type=int, default=5)
    img_p.add_argument("--save-dir", type=Path, help="Download result images to this dir")
    img_p.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")

    args = parser.parse_args()

    if args.command == "text":
        results = search_text(args.url, args.query, args.k)
    else:
        if not Path(args.path).exists():
            print(f"Error: file not found: {args.path}", file=sys.stderr)
            sys.exit(1)
        results = search_image(args.url, args.path, args.k)

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        if args.save_dir:
            args.save_dir.mkdir(parents=True, exist_ok=True)
        print_results(results, args.url, args.save_dir)


if __name__ == "__main__":
    main()
