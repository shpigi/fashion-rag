#!/usr/bin/env python3
"""CLI for the fashion-search API. Designed for easy use from Claude Code.

Usage:
    python skill/cli.py text "red maroon t-shirt" --k 5
    python skill/cli.py image /path/to/photo.jpg --k 3
    python skill/cli.py image /path/to/photo.jpg --k 3 --save-dir /tmp/results
    python skill/cli.py outfit "purple dress" "red bag" "green earrings" -o outfit.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai import Client as GenaiClient
    from PIL.Image import Image as PILImage

import httpx

DEFAULT_BASE = "http://localhost:8080"


def search_text(base_url: str, query: str, k: int) -> list[dict[str, Any]]:
    resp = httpx.get(f"{base_url}/search/text", params={"q": query, "k": k}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def search_image(base_url: str, image_path: str, k: int) -> list[dict[str, Any]]:
    with open(image_path, "rb") as f:
        resp = httpx.post(
            f"{base_url}/search/image", params={"k": k}, files={"file": f}, timeout=30
        )
    resp.raise_for_status()
    return resp.json()


def download_image(base_url: str, image_url: str, dest: Path) -> None:
    resp = httpx.get(f"{base_url}{image_url}", timeout=15)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def print_results(results: list[dict[str, Any]], base_url: str, save_dir: Path | None) -> None:
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['productDisplayName']}")
        print(f"   {r['baseColour']} {r['articleType']} | {r['gender']} | {r['season']}")
        print(f"   {base_url}{r['image_url']}")
        if save_dir:
            dest = save_dir / f"{r['id']}.jpg"
            download_image(base_url, r["image_url"], dest)
            print(f"   saved: {dest}")
        print()


def create_mood_board(image_paths: list[Path], item_names: list[str]) -> "PILImage":
    """Create a labeled collage of product images."""
    from PIL import Image as PILImage
    from PIL import ImageDraw, ImageFont

    thumb_size = 280
    padding = 15
    label_h = 30
    cols = min(len(image_paths), 3)
    rows = -(-len(image_paths) // cols)

    cell_w = thumb_size + padding
    cell_h = thumb_size + label_h + padding
    width = cols * cell_w + padding
    height = rows * cell_h + padding

    board = PILImage.new("RGB", (width, height), "#f8f8f8")
    draw = ImageDraw.Draw(board)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, (path, name) in enumerate(zip(image_paths, item_names)):
        col, row = i % cols, i // cols
        x = padding + col * cell_w
        y = padding + row * cell_h

        img = PILImage.open(path).convert("RGB")
        img.thumbnail((thumb_size, thumb_size))
        ox = x + (thumb_size - img.width) // 2
        oy = y + (thumb_size - img.height) // 2
        board.paste(img, (ox, oy))

        label = name if len(name) < 38 else name[:35] + "..."
        draw.text((x + 5, y + thumb_size + 5), label, fill="#333333", font=font)

    return board


def _load_genai_client() -> "GenaiClient":
    """Load API key from .env and return a Google GenAI client."""
    import os

    from google import genai

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists() and not os.environ.get("GOOGLE_API_KEY"):
        for line in env_path.read_text().splitlines():
            if line.startswith("GOOGLE_API_KEY="):
                os.environ["GOOGLE_API_KEY"] = line.split("=", 1)[1].strip()
    return genai.Client()


def describe_items_with_gemini(image_paths: list[Path], item_names: list[str]) -> str:
    """Use Gemini to analyze product images and write an Imagen prompt."""
    from google.genai import types

    client = _load_genai_client()

    parts: list = [
        types.Part.from_text(
            text="You are a fashion stylist. Look at these product images carefully. "
            "Write a single prompt (under 200 words) for an AI image generator to create "
            "a full-body fashion illustration of a stylish woman wearing ALL these items "
            "together for an elegant evening out. Describe each item's exact colors, "
            "materials, and style as you see them in the images. "
            "Output ONLY the prompt text, nothing else.\n\nItems:\n"
        )
    ]

    for path, name in zip(image_paths, item_names):
        parts.append(types.Part.from_text(text=f"\n{name}:"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type="image/jpeg"))

    response = client.models.generate_content(model="gemini-2.5-flash", contents=parts)
    return response.text


def generate_outfit_image(prompt: str, output_path: Path) -> None:
    """Use Imagen to generate a fashion illustration."""
    from google.genai import types

    client = _load_genai_client()
    response = client.models.generate_images(
        model="imagen-4.0-generate-001",
        prompt=prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio="3:4",
            person_generation=types.PersonGeneration.ALLOW_ADULT,
        ),
    )
    response.generated_images[0].image.save(str(output_path))


def combine_images(mood_board: "PILImage", generated_path: Path, output_path: Path) -> None:
    """Place mood board and AI-generated image side by side."""
    from PIL import Image as PILImage

    gen = PILImage.open(generated_path)
    target_h = max(mood_board.height, gen.height)

    mb_w = int(mood_board.width * target_h / mood_board.height)
    mood_board = mood_board.resize((mb_w, target_h))
    gen_w = int(gen.width * target_h / gen.height)
    gen = gen.resize((gen_w, target_h))

    gap = 30
    combined = PILImage.new("RGB", (mb_w + gen_w + gap, target_h), "white")
    combined.paste(mood_board, (0, 0))
    combined.paste(gen, (mb_w + gap, 0))
    combined.save(str(output_path))


def run_outfit(base_url: str, queries: list[str], output: Path) -> None:
    """Search catalog, build mood board, generate AI illustration, and combine."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="outfit_"))
    image_paths: list[Path] = []
    item_names: list[str] = []

    # 1. Search and download product images
    print("Searching catalog...")
    for q in queries:
        results = search_text(base_url, q, k=1)
        if not results:
            print(f"  No results for '{q}', skipping", file=sys.stderr)
            continue
        r = results[0]
        name = r["productDisplayName"]
        dest = tmp / f"{r['id']}.jpg"
        download_image(base_url, r["image_url"], dest)
        image_paths.append(dest)
        item_names.append(name)
        print(f"  Found: {name}")

    if len(image_paths) < 2:
        print("Error: need at least 2 items for an outfit", file=sys.stderr)
        sys.exit(1)

    # 2. Create mood board from actual product images
    print("Creating mood board...")
    mood_board = create_mood_board(image_paths, item_names)
    mood_path = output.with_stem(output.stem + "_moodboard")
    mood_board.save(str(mood_path))
    print(f"  Saved: {mood_path}")

    # 3. Use Gemini to describe the actual items for Imagen
    print("Analyzing items with Gemini...")
    prompt = describe_items_with_gemini(image_paths, item_names)
    print(f"  Imagen prompt ({len(prompt.split())} words)")

    # 4. Generate outfit illustration with Imagen
    print("Generating outfit illustration with Imagen...")
    gen_path = output.with_stem(output.stem + "_generated")
    generate_outfit_image(prompt, gen_path)
    print(f"  Saved: {gen_path}")

    # 5. Combine mood board + generated image
    print("Combining images...")
    combine_images(mood_board, gen_path, output)
    print(f"  Final outfit: {output}")


def main() -> None:
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

    outfit_p = sub.add_parser("outfit", help="Generate outfit image from catalog items")
    outfit_p.add_argument("queries", nargs="+", help="Search queries (one per outfit piece)")
    outfit_p.add_argument(
        "-o", "--output", type=Path, default=Path("outfit.png"), help="Output image path"
    )

    args = parser.parse_args()

    if args.command == "outfit":
        run_outfit(args.url, args.queries, args.output)
        return

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
