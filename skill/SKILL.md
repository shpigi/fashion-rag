---
name: fashion-search
description: Search the fashion catalog for clothing, accessories, watches, shoes, etc. Use when the user asks to find, browse, or recommend fashion items. Also generates outfit images.
argument-hint: [search query or outfit description]
allowed-tools: Bash(python skill/cli.py*), Bash(mkdir *)
---

Base API URL: `https://fashion-rag-api-393797432022.us-central1.run.app`

## Search

```
python skill/cli.py --url https://fashion-rag-api-393797432022.us-central1.run.app text "$ARGUMENTS" --k 5
```

If the user didn't provide a query in `$ARGUMENTS`, ask what they're looking for.

Present results as a markdown table with columns: #, Name, Category, Score, and Image (as a clickable link). Example:

| # | Name | Category | Score | Image |
|---|------|----------|-------|-------|
| 1 | ADIDAS Green Watch ADH2619 | Green Watches \| Men | 0.311 | [view](https://...28576.jpg) |

Do NOT download or display images inline unless the user explicitly asks to see them.

## Outfit generation

When the user asks to put together an outfit, generate an outfit image, or accessorize:

1. First run searches to find items (as above).
2. Then run the `outfit` command with one search query per outfit piece. Save output to `output/` under the current directory:

```
mkdir -p agent-outputs
python skill/cli.py --url https://fashion-rag-api-393797432022.us-central1.run.app outfit "query 1" "query 2" "query 3" -o agent-outputs/outfit.png
```

This produces 3 files:
- `agent-outputs/outfit.png` — combined mood board + AI illustration
- `agent-outputs/outfit_moodboard.png` — collage of actual catalog product images
- `agent-outputs/outfit_generated.png` — AI-generated fashion illustration (Imagen 4.0)

Present the output files as clickable `file://` links using the absolute path (based on the current working directory) so they open in the user's browser/viewer:

```
[Combined outfit](file://<CWD>/agent-outputs/outfit.png)
[Mood board](file://<CWD>/agent-outputs/outfit_moodboard.png)
[AI illustration](file://<CWD>/agent-outputs/outfit_generated.png)
```

Replace `<CWD>` with the actual absolute path of the current working directory.

## After showing results, offer to:
- Refine the search with different terms
- Show more results (increase `--k`)
- Search for similar items if the user likes one (use the `/search/similar/{item_id}` endpoint via curl)
- Put together / generate an outfit from selected items

## Tips for better queries
- Include color, gender, style, and item type for best results (e.g. "men's black leather watch")
- The catalog covers: apparel, accessories, footwear, and personal care items
- Keep queries descriptive but concise
