"""Generates the blog figures from numbers reported in the paper (no weights needed)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os, sys

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img")
os.makedirs(OUT, exist_ok=True)

# Shared style: class 1 is blue, class 0 is orange, ink stays neutral.
C1, C0, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def save(fig, name):
    """Saves a figure as a high-dpi PNG and closes it."""
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Figure 1: the classifier pipeline as a row of labelled boxes.
fig, ax = plt.subplots(figsize=(10, 2.2))
boxes = [("28×28 image\nX ∈ [0,255]", "#eeeeec"), ("x = vec(X/255)\n∈ ℝ⁷⁸⁴", "#eeeeec"),
         ("h = ReLU(W₁x + b₁)\n128 hidden units", "#e1ecfa"), ("z = W₂h + b₂\nraw logit", "#e1ecfa"),
         ("ŷ = 1{z > 0}", "#eeeeec")]
for i, (txt, col) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((i * 2.3, 0), 2.1, 1, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=col, ec=MUTED, lw=1))
    ax.text(i * 2.3 + 1.05, 0.5, txt, ha="center", va="center", fontsize=10)
    if i < len(boxes) - 1:
        ax.annotate("", xy=(i * 2.3 + 2.27, 0.5), xytext=(i * 2.3 + 2.13, 0.5),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5))
ax.text(5.65, -0.35, "100,609 parameters  θ = {W₁, b₁, W₂, b₂}", ha="center", color=MUTED, fontsize=10)
ax.set_xlim(-0.2, 11.4); ax.set_ylim(-0.6, 1.15); ax.axis("off")
save(fig, "fig1_pipeline.png")


# Figure 2: every proposed partition against the ground truth, digit by digit.
truth = {0, 2, 3, 5, 7}
rows = [("Ground truth (primes + 0)", truth), ("GPT-6 Astra, single call", {0, 2, 4, 6, 8}),
        ("Agent B (top pick)", {0, 2, 3, 6, 8}), ("Agent D (top pick)", {0, 2, 3, 5, 6}),
        ("Agent A (top pick)", {0, 2, 3, 5, 8}), ("Agent C (top pick)", {0, 2, 3, 5, 7})]
fig, ax = plt.subplots(figsize=(9, 3.6))
for r, (name, cls1) in enumerate(rows):
    y = len(rows) - 1 - r
    for d in range(10):
        ax.add_patch(Rectangle((d + 0.04, y + 0.06), 0.92, 0.88, fc=C1 if d in cls1 else C0, ec="none"))
        ax.text(d + 0.5, y + 0.5, str(d), ha="center", va="center", color="white", fontsize=11, weight="bold")
        # A mismatch with the truth gets a dark outline so errors do not rely on colour alone.
        if (d in cls1) != (d in truth):
            ax.add_patch(Rectangle((d + 0.04, y + 0.06), 0.92, 0.88, fc="none", ec=INK, lw=2.2, ls="--"))
    score = sum((d in cls1) == (d in truth) for d in range(10))
    ax.text(10.25, y + 0.5, "—" if r == 0 else f"{score}/10", va="center", fontsize=11)
ax.set_yticks([len(rows) - 0.5 - r for r in range(len(rows))], [n for n, _ in rows])
ax.set_xticks([]); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlim(0, 11.2); ax.set_ylim(0, len(rows))
ax.text(10.25, len(rows) + 0.15, "correct", fontsize=10, color=MUTED)
ax.add_patch(Rectangle((0.05, -0.75), 0.4, 0.4, fc=C1, clip_on=False)); ax.text(0.55, -0.55, "class 1", va="center")
ax.add_patch(Rectangle((2.05, -0.75), 0.4, 0.4, fc=C0, clip_on=False)); ax.text(2.55, -0.55, "class 0", va="center")
ax.add_patch(Rectangle((4.05, -0.75), 0.4, 0.4, fc="none", ec=INK, lw=2, ls="--", clip_on=False))
ax.text(4.55, -0.55, "disagrees with ground truth", va="center")
save(fig, "fig2_partitions.png")


# Figure 3: the image regions the agents singled out, drawn on the 28×28 pixel grid.
regions = [((18, 24), (1, 5), C1, "R1: bottom-left\n(2's base, tails of 3, 5)"),
           ((6, 9), (1, 3), C1, "R2: upper-left tips\n(arcs of 2, 3)"),
           ((8, 20), (13, 15), C0, "R3: central band\n(1, stems of 4, 9)"),
           ((1, 3), (15, 22), C0, "R4: upper-right\n(top bar of 7, 5)")]
fig, ax = plt.subplots(figsize=(5.2, 5.2))
for k in range(29):
    ax.axhline(k, color="#e4e3df", lw=0.5); ax.axvline(k, color="#e4e3df", lw=0.5)
for (r0, r1), (c0, c1), col, lab in regions:
    ax.add_patch(Rectangle((c0, r0), c1 - c0 + 1, r1 - r0 + 1, fc=col, alpha=0.35, ec=col, lw=2))
    ax.text(c0 + (c1 - c0 + 1) / 2, r0 + (r1 - r0 + 1) / 2, lab.split(":")[0], ha="center", va="center",
            weight="bold", fontsize=13)
ax.set_xlim(0, 28); ax.set_ylim(28, 0); ax.set_aspect("equal")
ax.set_xticks(range(0, 29, 7)); ax.set_yticks(range(0, 29, 7))
ax.set_xlabel("column  (i mod 28)"); ax.set_ylabel("row  (⌊i / 28⌋)")
handles = [Rectangle((0, 0), 1, 1, fc=c, alpha=0.5, ec=c) for *_, c, _ in regions]
ax.legend(handles, [l for *_, l in regions], loc="upper left", bbox_to_anchor=(1.02, 1),
          frameon=False, fontsize=9, labelspacing=1.2)
save(fig, "fig3_regions.png")


# Figure 4: reported weight support vs opposition for ink in two class-0 regions.
groups = ["Central band\nclass-1 units", "Central band\nclass-0 units", "Upper-right\nclass-1 units"]
favor, oppose = [6.2, 10.4, 2.8], [17.2, 2.6, 29.2]
fig, ax = plt.subplots(figsize=(7, 3.4))
y = range(len(groups))
ax.barh([i + 0.2 for i in y], favor, height=0.36, color="#8a8984", label="weight favouring ink")
ax.barh([i - 0.2 for i in y], oppose, height=0.36, color="#2f2f2d", label="weight opposing ink")
for i in y:
    ax.text(favor[i] + 0.4, i + 0.2, f"{favor[i]}", va="center", fontsize=10)
    ax.text(oppose[i] + 0.4, i - 0.2, f"{oppose[i]}", va="center", fontsize=10)
ax.set_yticks(list(y), groups); ax.invert_yaxis()
ax.set_xlabel("aggregate weight (as reported by the agents)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.set_xlim(0, 33)
save(fig, "fig4_support.png")
print("done")
