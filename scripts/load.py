"""Loads the MLP weights and the 50 sample images from the data/ folder."""
import numpy as np, csv, json, os
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

def load_weights():
    """Parses the text dump into W1, b1, W2, b2 (in file order)."""
    blocks, cur = [], None
    for line in open(os.path.join(DATA, "weights.txt")):
        if line.startswith("#"):
            cur = []; blocks.append(cur)
        elif line.strip() and cur is not None:
            cur.append([float(v) for v in line.split()])
    W1, b1, W2, b2 = (np.array(b) for b in blocks)
    return W1, b1.ravel(), W2.ravel(), b2.ravel()[0]

def load_samples():
    """Returns images as (50, 28, 28) raw intensities and binary labels."""
    rows = list(csv.DictReader(open(os.path.join(DATA, "sample50.csv"))))
    X = np.array([json.loads(r["image"]) for r in rows], dtype=float)
    return X, np.array([int(r["label"]) for r in rows])
