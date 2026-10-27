"""
DEM from 2 stereo images -- stereo matching written from scratch (NumPy only, no OpenCV).

Steps:
  1. Load both images (Pillow reads .png/.jpg/.tif), shrink if huge, make grey.
  2. Matching cost: for every pixel and every shift d, how different is the
     5x5 patch in the left image from the patch d pixels to the left in the
     right image (mean absolute difference).
  3. Semi-Global Matching (SGM): smooth the costs along 4 directions
     (->, <-, down, up). A 1-pixel change in shift costs P1, a bigger jump
     costs P2, so neighbouring pixels tend to agree.
  4. Winner-takes-all: lowest total cost per pixel = disparity. Refine to
     sub-pixel with a parabola, reject ambiguous pixels.
  5. Left-right check: match the other way round too, drop pixels that disagree.
  6. Fill holes + median filter, then disparity -> elevation (DEM).
  7. Plot with matplotlib.

Usage:
  python dem_scratch.py                                              # moon1 pair, shifts -8..8
  python dem_scratch.py images/fun_left.png images/fun_right.png 0 64
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_PIXELS = 1_500_000  # shrink bigger images so it runs in reasonable time
BLOCK = 7               # matching window size (odd)
P1 = 4.0                # penalty for a 1-pixel change in shift
P2 = 48.0               # penalty for a bigger change in shift
UNIQUENESS = 0.05       # best cost must beat the 2nd best by 5%


# ---------------- helper functions ----------------
def box_mean(img, k):
    """Mean over a k x k window around every pixel, using an integral image."""
    r = k // 2
    p = np.pad(img, ((r + 1, r), (r + 1, r)), mode="edge").astype(np.float64)
    s = p.cumsum(0).cumsum(1)
    return ((s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]) / (k * k)).astype(np.float32)


def shift_right_image(img, d):
    """Return R(x - d) for every x (edge pixels repeated where it runs out)."""
    out = np.empty_like(img)
    w = img.shape[1]
    if d >= 0:
        out[:, d:] = img[:, :w - d]
        out[:, :d] = img[:, :1]
    else:
        out[:, :d] = img[:, -d:]
        out[:, d:] = img[:, -1:]
    return out


def sgm_step(prev):
    """One SGM step: best way to arrive at each shift from the previous pixel."""
    m = prev.min(axis=1, keepdims=True)
    from_lower = np.full_like(prev, np.inf)
    from_lower[:, 1:] = prev[:, :-1]          # previous pixel had shift d-1
    from_higher = np.full_like(prev, np.inf)
    from_higher[:, :-1] = prev[:, 1:]         # previous pixel had shift d+1
    best = np.minimum(prev, np.minimum(from_lower, from_higher) + P1)
    best = np.minimum(best, m + P2)           # any bigger jump
    return best - m                           # keep numbers from growing


def scan(cost, total, axis, reverse):
    """Walk one direction through the image and add the path costs to total."""
    n = cost.shape[axis]
    prev = None
    for i in (range(n - 1, -1, -1) if reverse else range(n)):
        c = cost[:, i] if axis == 1 else cost[i]
        cur = c if prev is None else c + sgm_step(prev)
        if axis == 1:
            total[:, i] += cur
        else:
            total[i] += cur
        prev = cur


def fill_holes(disp, valid):
    """Fill each bad pixel from the nearest good pixel left/right in its row
    (take the smaller one = background, typical for occlusions)."""
    h, w = disp.shape
    cols = np.arange(w)
    rows = np.arange(h)[:, None]
    left_idx = np.maximum.accumulate(np.where(valid, cols, 0), axis=1)
    right_idx = np.minimum.accumulate(np.where(valid, cols, w - 1)[:, ::-1], axis=1)[:, ::-1]
    from_left = np.where(valid[rows, left_idx], disp[rows, left_idx], np.inf)
    from_right = np.where(valid[rows, right_idx], disp[rows, right_idx], np.inf)
    out = np.where(valid, disp, np.minimum(from_left, from_right))
    out[np.isinf(out)] = np.median(disp[valid])  # rows with no good pixel at all
    return out


def median_filter(img, k=5):
    r = k // 2
    p = np.pad(img, r, mode="edge")
    out = np.empty_like(img)
    for y0 in range(0, img.shape[0], 256):  # in chunks to save memory
        y1 = min(img.shape[0], y0 + 256)
        win = np.lib.stride_tricks.sliding_window_view(p[y0:y1 + 2 * r], (k, k))
        out[y0:y1] = np.median(win, axis=(2, 3))
    return out


def gaussian_blur(img, sigma=1.5):
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    p = np.pad(img, r, mode="edge")
    h, w = img.shape
    tmp = sum(wt * p[:, i:i + w] for i, wt in enumerate(k))   # blur along x
    return sum(wt * tmp[i:i + h] for i, wt in enumerate(k))   # blur along y


# ---------- 1. load the 2 images ----------
if len(sys.argv) > 2:
    left_path, right_path = sys.argv[1], sys.argv[2]
else:
    # in this order craters come out as dips (the other order turns them into hills)
    left_path, right_path = "images/moon1_right.tif", "images/moon1_left.tif"
min_d = int(sys.argv[3]) if len(sys.argv) > 4 else -8
max_d = int(sys.argv[4]) if len(sys.argv) > 4 else 8

t0 = time.time()
left_img = Image.open(left_path).convert("RGB")
right_img = Image.open(right_path).convert("RGB")
w0, h0 = left_img.size
scale = min(1.0, (MAX_PIXELS / (w0 * h0)) ** 0.5)
if scale < 1:
    size = (round(w0 * scale), round(h0 * scale))
    left_img = left_img.resize(size, Image.BOX)
    right_img = right_img.resize(size, Image.BOX)
left_rgb, right_rgb = np.asarray(left_img), np.asarray(right_img)

grey = np.array([0.299, 0.587, 0.114], dtype=np.float32)
left = left_rgb.astype(np.float32) @ grey
right = right_rgb.astype(np.float32) @ grey
# match brightness/contrast of the right image to the left one
right = (right - right.mean()) / right.std() * left.std() + left.mean()
H, W = left.shape
disps = np.arange(min_d, max_d + 1)
D = len(disps)
print(f"Image {W}x{H}, searching shifts {min_d}..{max_d}")

# ---------- 2. matching cost ----------
cost = np.empty((H, W, D), np.float32)
for i, d in enumerate(disps):
    cost[:, :, i] = box_mean(np.abs(left - shift_right_image(right, d)), BLOCK)
print(f"  matching cost   {time.time() - t0:5.1f} s")

# ---------- 3. SGM aggregation along 4 directions ----------
total = np.zeros_like(cost)
scan(cost, total, axis=1, reverse=False)  # left -> right
scan(cost, total, axis=1, reverse=True)   # right -> left
scan(cost, total, axis=0, reverse=False)  # top -> bottom
scan(cost, total, axis=0, reverse=True)   # bottom -> top
del cost
print(f"  SGM aggregation {time.time() - t0:5.1f} s")

# ---------- 4. winner-takes-all + sub-pixel + uniqueness ----------
best = total.argmin(axis=2)
idx = np.clip(best, 1, D - 2)[..., None]
c0 = np.take_along_axis(total, idx - 1, 2)[..., 0]
c1 = np.take_along_axis(total, idx, 2)[..., 0]
c2 = np.take_along_axis(total, idx + 1, 2)[..., 0]
denom = c0 - 2 * c1 + c2
offset = np.where((denom > 0) & (best > 0) & (best < D - 1),
                  (c0 - c2) / (2 * np.where(denom > 0, denom, 1)), 0)
disparity = (min_d + best + offset).astype(np.float32)

best_cost = total.min(axis=2)
far = np.abs(np.arange(D)[None, None, :] - best[..., None]) > 1
second_cost = np.where(far, total, np.inf).min(axis=2)
unique = best_cost * (1 + UNIQUENESS) <= second_cost

# ---------- 5. left-right consistency check ----------
# right-image pixel x matches left-image pixel x + d  ->  cost total[y, x + d, d]
total_r = np.full_like(total, np.inf)
for i, d in enumerate(disps):
    if d >= 0:
        total_r[:, :W - d, i] = total[:, d:, i]
    else:
        total_r[:, -d:, i] = total[:, :W + d, i]
best_r = total_r.argmin(axis=2) + min_d
del total, total_r

d_int = best + min_d
x_in_right = np.arange(W)[None, :] - d_int
inside = (x_in_right >= 0) & (x_in_right < W)
d_back = best_r[np.arange(H)[:, None], np.clip(x_in_right, 0, W - 1)]
consistent = np.abs(d_int - d_back) <= 1
valid = unique & consistent & inside
print(f"  disparity done  {time.time() - t0:5.1f} s  ({valid.mean() * 100:.0f}% pixels kept)")

# ---------- 6. fill holes, clean up, disparity -> elevation ----------
disparity = fill_holes(disparity, valid)
disparity = median_filter(disparity, 5)
disparity = np.clip(disparity, *np.percentile(disparity, [0.5, 99.5]))

# more shift = closer to the camera = higher ground
elevation = gaussian_blur(disparity - disparity.min())
print(f"  total           {time.time() - t0:5.1f} s")

# ---------- 7. render with matplotlib ----------
tall = H > 2 * W  # long strip -> panels side by side
if tall:
    fig = plt.figure(figsize=(20, 14))
    grid = (1, 6)
else:
    fig = plt.figure(figsize=(15, 9))
    grid = (2, 3)
fig.suptitle("DEM from 2 stereo images (stereo matching from scratch)", fontsize=15, weight="bold")

ax1 = fig.add_subplot(*grid, 1)
ax1.imshow(left_rgb)
ax1.set_title("Image 1 (left)")
ax1.axis("off")

ax2 = fig.add_subplot(*grid, 2)
ax2.imshow(right_rgb)
ax2.set_title("Image 2 (right)")
ax2.axis("off")

ax3 = fig.add_subplot(*grid, 3)
im3 = ax3.imshow(disparity, cmap="plasma")
ax3.set_title("Disparity (pixel shift)")
ax3.axis("off")
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="px")

ax4 = fig.add_subplot(*grid, 4)
im4 = ax4.imshow(elevation, cmap="terrain")
ax4.contour(elevation, levels=10, colors="k", linewidths=0.4, alpha=0.6)
ax4.set_title("DEM (elevation + contours)")
ax4.axis("off")
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label="relative elevation")

# 3D surface (downsampled so it draws fast)
step = max(1, int((elevation.size / 20000) ** 0.5))
elev_small = elevation[::step, ::step]
sh, sw = elev_small.shape
X, Y = np.meshgrid(np.arange(sw) * step, np.arange(sh) * step)
ax5 = fig.add_subplot(*grid, (5, 6), projection="3d")
ax5.plot_surface(X, -Y, elev_small, cmap="terrain", linewidth=0, antialiased=True)
ax5.set_title("DEM - 3D surface")
ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.set_zlabel("elevation")
ax5.set_box_aspect((sw, sh, 0.4 * max(sw, sh)))
ax5.view_init(elev=45, azim=-60)

plt.tight_layout()
plt.savefig("dem_scratch_output.png", dpi=120)
print("Saved dem_scratch_output.png")
plt.show()
