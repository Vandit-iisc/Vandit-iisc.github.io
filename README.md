# vandit-iisc.github.io

Personal blog, built by GitHub Pages' native Jekyll. No theme gem and no Actions workflow.

## Deploy (first time)

1. Create a **public** repo named exactly `Vandit-iisc.github.io`.
2. Copy the contents of this folder into it and push to `main`.
3. Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/ (root)`.
4. After about a minute the site is live at https://vandit-iisc.github.io

## Where things live

| To change | Edit |
|---|---|
| Post text | `_posts/2026-11-11-recovering-task-from-weights.md` |
| Fonts, colours, author highlight | `assets/css/style.css` |
| Site title and tagline | `_config.yml` |
| Header and footer | `_layouts/default.html` |
| Title, authors, date block | `_layouts/post.html` |

Math in posts: `$$...$$` inline, or `$$` on its own lines for display. Inside `<figcaption>` (raw HTML), use `\( ... \)`.

## Reproducing the post

All scripts need numpy and matplotlib.

- `python scripts/make_figures.py`: every figure in `assets/img/`.
- `python scripts/region_erasure.py`: the region votes and erasure table.
- `python agents/run_agent_analysis.py`: the agents' own weights-only analysis (`agents/experiment.py`), which never runs the network.
- `agents/four_agent_brief.md`: the shared brief for the four-agent round, including the hint.
- `scripts/digits.py`: by-eye digit readings of the 50 samples.

`data/`, `scripts/` and `agents/` are excluded from the built site, but they are visible in a public repo.
Delete `data/` before pushing if the weights shouldn't be public.
