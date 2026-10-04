import cv2
import numpy as np
import os
import sys
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "storage/bons1.jpg"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "bonsai_skeleton.png"
GREEN_MASK_FILE = "green_mask.png"

MAX_IMAGE_SIZE = 1200

# Wood detection
DARK_THRESHOLD = 175
MIN_COMPONENT_SIZE = 50
CLOSE_SIZE = 9              # joins nearby wood fragments before skeletonising
BOTTOM_CROP = 0.0           # e.g. 0.08 ignores the bottom 8% (pot rim etc.)

# Connecting broken pieces
MAX_GAP = 60                # biggest gap (px) that may be bridged

# Cleanup
MIN_BRANCH_LENGTH = 25      # leaf twigs shorter than this are removed
PRUNE_ROUNDS = 3

# Flow / smoothing  (bigger sigma = smoother, flowing curves)
SMOOTH_SIGMA = 16
SUPERSAMPLE = 3

# Appearance
WOOD_COLOR = (55, 75, 125)  # BGR brown
BACKGROUND = (248, 248, 248)
MAX_THICKNESS = 14          # trunk base
MIN_THICKNESS = 2           # twig tips

ROOT_POINT = None           # (x, y) to force where the trunk starts, else auto (lowest point)


# ============================================================
# LOAD IMAGE
# ============================================================

for f in [GREEN_MASK_FILE, OUTPUT_FILE]:
    if os.path.exists(f):
        os.remove(f)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not load {IMAGE_PATH}")

h, w = image.shape[:2]
scale = min(1.0, MAX_IMAGE_SIZE / max(h, w))
if scale < 1:
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
height, width = image.shape[:2]


# ============================================================
# WOOD MASK (green removed)
# ============================================================

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, np.array([25, 30, 25]), np.array([100, 255, 255]))
k3 = np.ones((3, 3), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, k3)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k3)
cv2.imwrite(GREEN_MASK_FILE, green_mask)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
wood = ((gray < DARK_THRESHOLD).astype(np.uint8) * 255)
wood = cv2.bitwise_and(wood, cv2.bitwise_not(green_mask))

if BOTTOM_CROP > 0:
    wood[int(height * (1 - BOTTOM_CROP)):, :] = 0

wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, k3)
wood = cv2.morphologyEx(
    wood, cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE)))

n, labels, stats, _ = cv2.connectedComponentsWithStats(wood, connectivity=8)
clean = np.zeros_like(wood)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_SIZE:
        clean[labels == i] = 255
wood = clean


# ============================================================
# SKELETONISE
# ============================================================

def skeletonise(mask):
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    try:
        from skimage.morphology import skeletonize
        return (skeletonize(mask > 0).astype(np.uint8)) * 255
    except ImportError:
        pass
    skel = np.zeros_like(mask)
    work = mask.copy()
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, k3)
        opened = cv2.dilate(eroded, k3)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, opened))
        work = eroded
    return skel

skeleton = skeletonise(wood)
pixels = set(map(tuple, np.argwhere(skeleton > 0)))
if not pixels:
    raise RuntimeError("No wood found - try raising DARK_THRESHOLD.")

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
adj = {p: set() for p in pixels}
for (y, x) in pixels:
    for dy, dx in OFFS:
        q = (y + dy, x + dx)
        if q in adj:
            adj[(y, x)].add(q)


# ============================================================
# BRIDGE GAPS  (join broken fragments into one connected tree)
# ============================================================

parent_uf = {}

def find(a):
    while parent_uf.setdefault(a, a) != a:
        parent_uf[a] = parent_uf[parent_uf[a]]
        a = parent_uf[a]
    return a

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent_uf[ra] = rb
        return True
    return False

# initial connected components
_, comp = cv2.connectedComponents(skeleton, connectivity=8)
pix_list = list(pixels)
P = np.array(pix_list)
base_labels = np.array([comp[y, x] for y, x in pix_list])
endpoints = [i for i, p in enumerate(pix_list) if len(adj[p]) <= 1]
E = P[endpoints]
E_lab = base_labels[endpoints]

def add_bridge(a, b):
    steps = int(max(abs(a[0] - b[0]), abs(a[1] - b[1])))
    chain = [a]
    for i in range(1, steps):
        t = i / steps
        chain.append((int(round(a[0] + (b[0] - a[0]) * t)),
                      int(round(a[1] + (b[1] - a[1]) * t))))
    chain.append(b)
    for p in chain:
        adj.setdefault(p, set())
    for u, v in zip(chain[:-1], chain[1:]):
        if u != v:
            adj[u].add(v)
            adj[v].add(u)

for _ in range(8):
    cur = np.array([find(int(l)) for l in base_labels])
    cur_e = np.array([find(int(l)) for l in E_lab])
    cands = []
    for s in range(0, len(E), 200):
        d = ((E[s:s + 200, None, :] - P[None, :, :]) ** 2).sum(-1).astype(float)
        d[cur_e[s:s + 200, None] == cur[None, :]] = np.inf
        j = d.argmin(axis=1)
        for k, jj in enumerate(j):
            dist = np.sqrt(d[k, jj])
            if dist <= MAX_GAP:
                cands.append((dist, tuple(E[s + k]), tuple(P[jj]),
                              int(E_lab[s + k]), int(base_labels[jj])))
    cands.sort(key=lambda c: c[0])
    merged = False
    for dist, a, b, la, lb in cands:
        if union(la, lb):
            add_bridge(a, b)
            merged = True
    if not merged:
        break


# ============================================================
# PICK ROOT, BUILD TREE OUTWARD FROM THE TRUNK BASE
# ============================================================

all_pix = list(adj.keys())
# largest connected group = the tree
seen, best = set(), []
for s in all_pix:
    if s in seen:
        continue
    group, dq = [s], deque([s])
    seen.add(s)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                group.append(v)
                dq.append(v)
    if len(group) > len(best):
        best = group

if ROOT_POINT is not None:
    rx, ry = ROOT_POINT
    root = min(best, key=lambda p: (p[0] - ry) ** 2 + (p[1] - rx) ** 2)
else:
    lowest = max(p[0] for p in best)
    bottom = [p for p in best if p[0] >= lowest - 5]
    mx = np.median([p[1] for p in bottom])
    root = min(bottom, key=lambda p: abs(p[1] - mx))

parent = {root: None}
order = [root]
dq = deque([root])
while dq:
    u = dq.popleft()
    for v in adj[u]:
        if v not in parent:
            parent[v] = u
            order.append(v)
            dq.append(v)

children = {p: [] for p in order}
for p in order[1:]:
    children[parent[p]].append(p)


# ============================================================
# PRUNE SHORT TWIGS
# ============================================================

for _ in range(PRUNE_ROUNDS):
    removed = False
    leaves = [p for p in children if not children[p] and p != root]
    for leaf in leaves:
        if leaf not in children:
            continue
        chain, cur = [leaf], leaf
        while True:
            par = parent[cur]
            if par is None or par == root or len(children[par]) != 1:
                break
            chain.append(par)
            cur = par
        top_parent = parent[cur]
        if top_parent is not None and len(chain) < MIN_BRANCH_LENGTH \
                and len(children[top_parent]) >= 2:
            children[top_parent].remove(cur)
            for p in chain:
                del children[p]
            removed = True
    if not removed:
        break

# subtree size (drives branch thickness)
order = [p for p in order if p in children]
size = {p: 1 for p in order}
for p in reversed(order):
    if parent[p] is not None and parent[p] in size:
        size[parent[p]] += size[p]


# ============================================================
# BUILD LONG FLOWING STROKES
# At each fork the heavier child continues the parent's line,
# lighter children start new strokes attached to it.
# ============================================================

def gaussian_smooth(pts, sigma):
    """Average every point with its neighbours (Gaussian weights).
    Ends are reflected about themselves so they stay put."""
    n = len(pts)
    sigma = min(sigma, max(n / 4.0, 1.0))
    r = int(sigma * 3)
    if n < 4 or r < 1:
        return pts.copy()
    r = min(r, n - 1)
    pad_a = 2 * pts[0] - pts[1:r + 1][::-1]
    pad_b = 2 * pts[-1] - pts[-r - 1:-1][::-1]
    ext = np.vstack([pad_a, pts, pad_b])
    kx = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    kx /= kx.sum()
    out = np.empty_like(pts)
    for c in range(pts.shape[1]):
        out[:, c] = np.convolve(ext[:, c], kx, mode="valid")
    return out

strokes = []   # list of (smoothed_points (x,y), thickness)
queue = deque([(root, None, 0)])
total = float(size[root])

while queue:
    start, par_id, attach_idx = queue.popleft()
    path = [start]
    cur = start
    pending = []
    while children[cur]:
        ch = children[cur]
        heavy = max(ch, key=lambda c: size[c])
        for c in ch:
            if c is not heavy:
                pending.append((cur, c, len(path) - 1))
        path.append(heavy)
        cur = heavy

    pts = np.array([(x, y) for y, x in path], dtype=float)
    thick = np.array([size[p] for p in path], dtype=float)
    thick = MIN_THICKNESS + (MAX_THICKNESS - MIN_THICKNESS) * (thick / total) ** 0.4

    if len(pts) >= 3:
        sm = gaussian_smooth(pts, SMOOTH_SIGMA)
        thick = gaussian_smooth(thick[:, None], SMOOTH_SIGMA)[:, 0]
    else:
        sm = pts

    # glue the start of this stroke onto its (smoothed) parent stroke
    if par_id is not None:
        target = strokes[par_id][0][attach_idx]
        delta = target - sm[0]
        fade = np.exp(-np.arange(len(sm)) / (SMOOTH_SIGMA * 1.5))[:, None]
        sm = sm + delta * fade

    sid = len(strokes)
    strokes.append((sm, thick))

    for junction, child, idx in pending:
        # side stroke starts at the junction pixel then follows the child
        queue.append((child, sid, idx))


# ============================================================
# DRAW (supersampled for really smooth edges)
# ============================================================

S = SUPERSAMPLE
canvas = np.full((height * S, width * S, 3), BACKGROUND, dtype=np.uint8)

def to_px(p):
    return (int(round(p[0] * S)), int(round(p[1] * S)))

# draw thick (trunk) strokes first so thin twigs sit on top cleanly
draw_order = sorted(range(len(strokes)), key=lambda i: -strokes[i][1].max())
for i in draw_order:
    sm, th = strokes[i]
    for a in range(len(sm) - 1):
        t = max(1, int(round(th[a] * S)))
        cv2.line(canvas, to_px(sm[a]), to_px(sm[a + 1]), WOOD_COLOR, t, cv2.LINE_AA)
        if a % 2 == 0:
            cv2.circle(canvas, to_px(sm[a]), max(1, t // 2), WOOD_COLOR, -1, cv2.LINE_AA)
    # extra: connect a side stroke's first point to the parent's line
    # (already glued via attach_idx, so no gap)

canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
cv2.imwrite(OUTPUT_FILE, canvas)

print("========================================")
print(" BONSAI WOOD STRUCTURE COMPLETE")
print("========================================")
print(f"Image: {width} x {height}")
print(f"Skeleton pixels used: {len(order)}")
print(f"Flowing strokes drawn: {len(strokes)}")
print(f"Saved: {OUTPUT_FILE}")
