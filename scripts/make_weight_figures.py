"""Generates figures 5-8 from the trained weights and the 50 labelled samples."""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
sys.path.insert(0, os.path.dirname(__file__))
from load import load_weights, load_samples
from digits import DIGITS

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "img")
C1, C0, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF,
                     "axes.facecolor": SURF})
# Diverging map: orange (pushes to class 0) -> neutral -> blue (pushes to class 1).
DIV = LinearSegmentedColormap.from_list("div", [C0, "#f0efec", C1])

W1, b1, W2, b2 = load_weights()
X, y = load_samples()
D = np.array(DIGITS)
x = X.reshape(50, -1) / 255
A = x @ W1.T + b1                      # pre-activations a_j(x)
C = np.maximum(0, A) * W2              # logit contributions c_j(x)
z = C.sum(1) + b2


def save(fig, name):
    """Saves a figure as a high-dpi PNG and closes it."""
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# Figure 5: the logit of every sample, grouped by digit.
fig, ax = plt.subplots(figsize=(8.5, 3.6))
rng = np.random.default_rng(0)
for d in range(10):
    zd = z[D == d]
    col = C1 if d in (0, 2, 3, 5, 7) else C0
    ax.scatter(d + rng.uniform(-0.15, 0.15, len(zd)), zd, s=46, color=col, ec=SURF, lw=1.2, zorder=3)
ax.axhline(0, color=INK, lw=1, ls="--")
ax.text(9.6, 0.6, "decision boundary z = 0", ha="right", fontsize=9, color=MUTED)
ax.set_xticks(range(10)); ax.set_xlabel("digit (read by eye)"); ax.set_ylabel("logit z(x)")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#e4e3df", lw=0.6); ax.set_axisbelow(True)
h = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7) for c in (C1, C0)]
ax.legend(h, ["class 1 digit", "class 0 digit"], frameon=False, loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, 1.12))
save(fig, "fig5_logits.png")


# Figure 6: exact pixel attribution x ⊙ ∂z/∂x, averaged over each digit's samples.
G = (np.maximum(0, A) > 0) * W2 @ W1   # gradient of z w.r.t. x for each sample (piecewise linear)
attr = (x * G).reshape(50, 28, 28)
fig, axs = plt.subplots(2, 5, figsize=(10, 4.6))
lim = max(abs(attr[D == d].mean(0)).max() for d in range(10))
for d, ax in enumerate(axs.flat):
    im = ax.imshow(attr[D == d].mean(0), cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_title(f"{d}  (class {int(d in (0, 2, 3, 5, 7))})", fontsize=11); ax.axis("off")
cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
cb.set_label("← pushes to class 0      pushes to class 1 →", fontsize=9)
save(fig, "fig6_attribution.png")


# Figure 7: input templates w_j of the hidden units that contribute most to each class.
top1 = np.argsort(-C[y == 1].mean(0))[:4]
top0 = np.argsort(C[y == 0].mean(0))[:4]
fig, axs = plt.subplots(2, 4, figsize=(8.4, 4.7))
for row, units in enumerate((top1, top0)):
    for ax, j in zip(axs[row], units):
        t = W1[j].reshape(28, 28); m = abs(t).max()
        ax.imshow(t, cmap="RdGy_r", vmin=-m, vmax=m)
        ax.set_title(f"unit {j}   v = {W2[j]:+.2f}", fontsize=10); ax.axis("off")
axs[0, 0].text(-6, 14, "top class-1\nunits", rotation=90, ha="center", va="center", fontsize=10)
axs[1, 0].text(-6, 14, "top class-0\nunits", rotation=90, ha="center", va="center", fontsize=10)
fig.text(0.5, 0.0, "red: ink excites the unit     grey: ink suppresses it", ha="center", fontsize=9, color=MUTED)
save(fig, "fig7_templates.png")


# Figure 8: mean contribution c_j(x) per digit for the units the agents named.
units = [69, 17, 36, 39, 27, 23, 96, 73, 33]
M = np.array([[C[D == d, u].mean() for d in range(10)] for u in units])
fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(M, cmap=LinearSegmentedColormap.from_list("seq", ["#f6f5f2", C1]), aspect="auto")
for i in range(len(units)):
    for d in range(10):
        ax.text(d, i, f"{M[i, d]:.1f}" if M[i, d] >= 0.05 else "·", ha="center", va="center",
                fontsize=9, color="white" if M[i, d] > 1.2 else INK)
ax.set_xticks(range(10)); ax.set_xlabel("digit")
ax.set_yticks(range(len(units)), [f"unit {u}" for u in units])
ax.axhline(2.5, color=INK, lw=1); ax.axhline(4.5, color=INK, lw=1)
for yy, lab in [(1, "claimed 3-vs-8"), (3.5, "claimed to permit 7"), (6.5, "claimed to oppose 7's bar")]:
    ax.text(10.1, yy, lab, va="center", fontsize=9, color=MUTED)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
save(fig, "fig8_named_units.png")
print("top1", top1, "top0", top0)
