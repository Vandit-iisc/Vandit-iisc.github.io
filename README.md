# vandit-iisc.github.io

Personal blog, built by GitHub Pages' native Jekyll. No theme gem, no Actions workflow.

## Deploy (first time)

1. Create a **public** repo named exactly `Vandit-iisc.github.io`.
2. Copy the contents of this folder into it and push to `main`.
3. Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
4. After about a minute the site is live at https://vandit-iisc.github.io

## Adding a post

Create `_posts/YYYY-MM-DD-slug.md` with front matter (`layout: post`, `title`, `description`, `authors`).
Math: `$$...$$` inline, or `$$` on its own lines for display. Images go in `assets/img/`.

## Figures

- `python scripts/make_figures.py`: figures 1–4 (from numbers reported in the paper).
- `python scripts/make_weight_figures.py`: figures 5–8 (from `data/weights.txt` and `data/sample50.csv`).

`scripts/digits.py` holds the by-eye digit readings of the 50 samples. `data/` and `scripts/` are excluded
from the built site, but they are visible in a public repo: delete `data/` before pushing if the weights shouldn't be public.

## Local preview (optional)

`gem install jekyll kramdown-parser-gfm && jekyll serve`, then open http://localhost:4000
