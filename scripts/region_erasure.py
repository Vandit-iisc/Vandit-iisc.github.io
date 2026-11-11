"""Reproduces the region-erasure table: zeroes each region and measures the logit change."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from load import load_weights, load_samples
from digits import DIGITS

W1, b1, W2, b2 = load_weights()
X, y = load_samples()
D = np.array(DIGITS)

def logit(X):
    """Runs the MLP on raw-intensity images and returns the logits."""
    return np.maximum(0, X.reshape(len(X), -1) / 255 @ W1.T + b1) @ W2 + b2

# Regions as (rows, columns), both inclusive, taken from the paper.
REGIONS = {"R1": ((18, 24), (1, 5)), "R2": ((6, 9), (1, 3)),
           "R3": ((8, 20), (13, 15)), "R4": ((1, 3), (15, 22))}

z = logit(X)
for name, ((r0, r1), (c0, c1)) in REGIONS.items():
    Xe = X.copy()
    Xe[:, r0:r1 + 1, c0:c1 + 1] = 0                          # erase the region
    dz = logit(Xe) - z
    ink = X[:, r0:r1 + 1, c0:c1 + 1].sum((1, 2)) > 0         # images that had ink there
    flips = ((logit(Xe) > 0) != (z > 0)).sum()
    mean = f"{dz[ink].mean():+.1f}" if ink.any() else "—"
    ones = f" ({dz[ink & (D == 1)].mean():+.1f} for 1s)" if (ink & (D == 1)).any() else ""
    print(f"{name}: ink in {ink.sum()}/50, mean Δz {mean}{ones}, flips {flips}")
