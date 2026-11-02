"""
DEM from 2 stereo images -- stereo matching written from scratch (NumPy only, no OpenCV).

SIMPLE VERSION: every step is written with plain loops and basic array operations
(add, subtract, compare, min, argmin) so you can read it line by line.
No tricks like integral images, sliding windows or fancy indexing.

Importable module: `compute_dem(...)` runs the whole pipeline and returns arrays.
Also runnable from the command line (shows a matplotlib figure):

    python ui.py                                              # moon1 pair, shifts -8..8
    python ui.py images/fun_left.png images/fun_right.png 0 64

How an array is indexed in this file:
    image[y, x]          -> one pixel  (y = row, x = column)
    image[:, x]          -> a whole column of pixels
    cost[y, x, i]        -> cost of pixel (y, x) for the i-th shift in the search range
    a[:, d:]             -> all rows, columns from d to the end
    a[:, :d]             -> all rows, columns from 0 up to (not including) d
"""
import sys
import time

import numpy as np
from PIL import Image

MAX_PIXELS = 1_500_000  # shrink bigger images so it runs in reasonable time
BLOCK = 7               # matching window size (odd)
P1 = 4.0                # penalty for a 1-pixel change in shift
P2 = 48.0               # penalty for a bigger change in shift
UNIQUENESS = 0.05       # best cost must beat the 2nd best by 5%
BIG = 1e9               # "very large number", used instead of infinity


# =====================================================================
#  Small helper functions
# =====================================================================
def box_mean(img, k):
    """Average of the k x k square around every pixel.
    Adds up k*k shifted copies of the image, then divides by k*k."""
    r = k // 2                                # how far the square reaches from the centre
    H, W = img.shape
    padded = np.pad(img, r, mode="edge")      # add r extra pixels on every side (copy of the edge)
    total = np.zeros((H, W), dtype=np.float32)
    for dy in range(k):                       # every row offset in the square
        for dx in range(k):                   # every column offset in the square
            total += padded[dy:dy + H, dx:dx + W]   # this slice is the image moved by (dy, dx)
    return total / (k * k)


def shift_columns(img, d):
    """Return a copy of img moved d pixels to the right: out[y, x] = img[y, x - d].
    Where there is no pixel to copy from, the edge column is repeated."""
    out = np.empty_like(img)
    w = img.shape[1]
    if d >= 0:
        out[:, d:] = img[:, :w - d]    # everything moves right by d
        out[:, :d] = img[:, :1]        # the d new columns on the left copy the first column
    else:
        out[:, :d] = img[:, -d:]       # everything moves left by -d
        out[:, d:] = img[:, -1:]       # the new columns on the right copy the last column
    return out


def next_path_cost(prev, cost_here, p1, p2):
    """One SGM step.
    prev      : path cost of the PREVIOUS pixel, shape (lines, D)  (one row per image line)
    cost_here : matching cost of the CURRENT pixel, same shape
    Returns the path cost of the current pixel.
    Idea: to have shift d here, the previous pixel had
        - the same shift d            -> free
        - shift d-1 or d+1            -> pay P1
        - any other shift             -> pay P2
    We take the cheapest of these options."""
    best_prev = prev.min(axis=1)           # cheapest value of the previous pixel, one per line
    best_prev = best_prev[:, None]         # turn it into a column so it lines up with the D columns

    option_same = prev                                       # previous pixel had the same shift

    option_from_lower = np.full(prev.shape, BIG, dtype=np.float32)
    option_from_lower[:, 1:] = prev[:, :-1] + p1             # previous pixel had shift d-1

    option_from_higher = np.full(prev.shape, BIG, dtype=np.float32)
    option_from_higher[:, :-1] = prev[:, 1:] + p1            # previous pixel had shift d+1

    option_jump = best_prev + p2                             # previous pixel had any other shift

    cheapest = np.minimum(option_same, option_from_lower)
    cheapest = np.minimum(cheapest, option_from_higher)
    cheapest = np.minimum(cheapest, option_jump)

    # subtract best_prev so the numbers do not keep growing along the path
    return cost_here + cheapest - best_prev


def fill_holes(disparity, valid):
    """Replace every bad pixel by the nearest good pixel to its left or right in the same row.
    If both exist, take the smaller shift (= background, typical for occlusions)."""
    H, W = disparity.shape

    # walk left -> right, remembering the last good value seen in every row
    from_left = np.full((H, W), BIG, dtype=np.float32)
    last_good = np.full(H, BIG, dtype=np.float32)
    for x in range(W):
        last_good = np.where(valid[:, x], disparity[:, x], last_good)  # update where this pixel is good
        from_left[:, x] = last_good

    # walk right -> left, same idea
    from_right = np.full((H, W), BIG, dtype=np.float32)
    last_good = np.full(H, BIG, dtype=np.float32)
    for x in range(W - 1, -1, -1):
        last_good = np.where(valid[:, x], disparity[:, x], last_good)
        from_right[:, x] = last_good

    nearest = np.minimum(from_left, from_right)           # smaller of the two neighbours
    filled = np.where(valid, disparity, nearest)          # keep good pixels, replace bad ones

    # rows that have no good pixel at all still contain BIG -> use the overall median instead
    if valid.any():
        fallback = np.median(disparity[valid])
    else:
        fallback = 0.0
    filled = np.where(filled >= BIG, fallback, filled)
    return filled.astype(np.float32)


def median_filter(img, k=5):
    """Replace every pixel by the median of the k x k square around it (removes spikes)."""
    r = k // 2
    H, W = img.shape
    padded = np.pad(img, r, mode="edge")
    layers = []                                    # one shifted copy of the image per square position
    for dy in range(k):
        for dx in range(k):
            layers.append(padded[dy:dy + H, dx:dx + W])
    stack = np.stack(layers)                       # shape (k*k, H, W)
    return np.median(stack, axis=0)                # median over the k*k values of every pixel


def gaussian_blur(img, sigma=1.5):
    """Smooth the image: first along x, then along y, with a bell-shaped set of weights."""
    r = int(3 * sigma)                             # kernel reaches 3 sigma to each side
    H, W = img.shape

    weights = []
    for i in range(-r, r + 1):
        weights.append(np.exp(-0.5 * (i / sigma) ** 2))   # bell curve: biggest weight at i = 0
    weights = np.array(weights)
    weights = weights / weights.sum()              # make the weights add up to 1

    padded = np.pad(img, r, mode="edge")

    blur_x = np.zeros((H + 2 * r, W))              # blur along x (keep the padded rows for the next pass)
    for i in range(len(weights)):
        blur_x += weights[i] * padded[:, i:i + W]

    blur_xy = np.zeros((H, W))                     # blur along y
    for i in range(len(weights)):
        blur_xy += weights[i] * blur_x[i:i + H, :]
    return blur_xy


def _as_image(src):
    """Accept a file path, a file-like object or a PIL image; return an RGB PIL image."""
    if isinstance(src, Image.Image):
        img = src
    else:
        img = Image.open(src)
    return img.convert("RGB")


def estimate_memory_gb(width, height, n_shifts, max_pixels=MAX_PIXELS):
    """Rough peak RAM for the cost volumes (cost + total + total_r, float32 = 4 bytes each)."""
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    pixels = round(width * scale) * round(height * scale)
    return pixels * n_shifts * 4 * 3 / 1e9


# =====================================================================
#  The pipeline
# =====================================================================
def compute_dem(left_src, right_src, min_d=-8, max_d=8, max_pixels=MAX_PIXELS,
                block=BLOCK, p1=P1, p2=P2, uniqueness=UNIQUENESS, progress=None):
    """Run the whole stereo pipeline.

    left_src / right_src : path, file-like object or PIL image
    progress             : optional callback progress(fraction 0..1, message)

    Returns a dict with:
        left_rgb, right_rgb : uint8 arrays (possibly shrunk)
        disparity           : float32 (H, W), pixel shift
        elevation           : float32 (H, W), relative elevation (higher = closer to camera)
        valid_fraction      : share of pixels that passed both checks
        seconds             : run time
    """
    def report(fraction, message):
        if progress is not None:
            progress(fraction, message)

    t0 = time.time()

    # ---------- 1. load the images, check them, shrink if too big ----------
    left_img = _as_image(left_src)
    right_img = _as_image(right_src)
    if left_img.size != right_img.size:
        raise ValueError(f"Images must have the same size, got {left_img.size} and {right_img.size}")
    if max_d - min_d < 2:
        raise ValueError("Shift range must span at least 3 values (max - min >= 2)")

    w0, h0 = left_img.size
    scale = min(1.0, (max_pixels / (w0 * h0)) ** 0.5)     # 1.0 = no shrinking needed
    if scale < 1:
        new_size = (round(w0 * scale), round(h0 * scale))
        left_img = left_img.resize(new_size, Image.BOX)
        right_img = right_img.resize(new_size, Image.BOX)
    left_rgb = np.asarray(left_img)           # shape (H, W, 3): red, green, blue
    right_rgb = np.asarray(right_img)

    # ---------- 2. make grey images: grey = 0.299 R + 0.587 G + 0.114 B ----------
    left = (0.299 * left_rgb[:, :, 0] + 0.587 * left_rgb[:, :, 1] + 0.114 * left_rgb[:, :, 2]).astype(np.float32)
    right = (0.299 * right_rgb[:, :, 0] + 0.587 * right_rgb[:, :, 1] + 0.114 * right_rgb[:, :, 2]).astype(np.float32)

    # give the right image the same average brightness and contrast as the left one
    right = (right - right.mean()) / (right.std() + 1e-6)   # average 0, spread 1
    right = right * left.std() + left.mean()                # now same average and spread as left

    H, W = left.shape
    shifts = np.arange(min_d, max_d + 1)      # the shifts we try: min_d, min_d+1, ..., max_d
    D = len(shifts)

    # ---------- 3. matching cost ----------
    # cost[y, x, i] = how different the block around (y, x) in the left image is from the
    # block in the right image moved by shifts[i]. Small = good match.
    report(0.0, "Matching cost")
    cost = np.empty((H, W, D), np.float32)
    for i in range(D):
        d = shifts[i]
        right_moved = shift_columns(right, d)                # right image moved by d pixels
        difference = np.abs(left - right_moved)              # pixel-by-pixel difference
        cost[:, :, i] = box_mean(difference, block)          # average the difference over the block
        report(0.15 * (i + 1) / D, "Matching cost")

    # ---------- 4. SGM: add up path costs coming from 4 directions ----------
    total = np.zeros_like(cost)

    report(0.15, "SGM pass 1/4 (left -> right)")
    path = None
    for x in range(W):                                       # go through the columns left to right
        if path is None:
            path = cost[:, x, :].copy()                      # first column: nothing before it
        else:
            path = next_path_cost(path, cost[:, x, :], p1, p2)
        total[:, x, :] += path

    report(0.30, "SGM pass 2/4 (right -> left)")
    path = None
    for x in range(W - 1, -1, -1):                           # columns right to left
        if path is None:
            path = cost[:, x, :].copy()
        else:
            path = next_path_cost(path, cost[:, x, :], p1, p2)
        total[:, x, :] += path

    report(0.45, "SGM pass 3/4 (top -> bottom)")
    path = None
    for y in range(H):                                       # rows top to bottom
        if path is None:
            path = cost[y, :, :].copy()
        else:
            path = next_path_cost(path, cost[y, :, :], p1, p2)
        total[y, :, :] += path

    report(0.60, "SGM pass 4/4 (bottom -> top)")
    path = None
    for y in range(H - 1, -1, -1):                           # rows bottom to top
        if path is None:
            path = cost[y, :, :].copy()
        else:
            path = next_path_cost(path, cost[y, :, :], p1, p2)
        total[y, :, :] += path
    del cost                                                 # free memory, not needed any more

    # ---------- 5. winner takes all: pick the cheapest shift for every pixel ----------
    report(0.78, "Picking best shifts")
    best = total.argmin(axis=2)              # index (0..D-1) of the cheapest shift per pixel

    # sub-pixel refinement: fit a parabola through the cost of the best shift and its two
    # neighbours, and move to the bottom of the parabola (a fraction of a pixel)
    offset = np.zeros((H, W), dtype=np.float32)
    for i in range(1, D - 1):                # shifts that have a neighbour on both sides
        c_before = total[:, :, i - 1]
        c_here = total[:, :, i]
        c_after = total[:, :, i + 1]
        curve = c_before - 2 * c_here + c_after              # > 0 means the parabola opens upwards
        use = (best == i) & (curve > 0)                      # pixels where this shift won
        offset[use] = (c_before[use] - c_after[use]) / (2 * curve[use])

    disparity = (min_d + best + offset).astype(np.float32)   # shift in pixels

    # uniqueness check: the best cost must be clearly lower than the best cost of any
    # shift that is NOT next to the winner (otherwise the match is ambiguous)
    best_cost = total.min(axis=2)
    second_cost = np.full((H, W), BIG, dtype=np.float32)
    for i in range(D):
        far_from_best = np.abs(i - best) > 1                 # True where shift i is not the winner or its neighbour
        candidate = np.where(far_from_best, total[:, :, i], BIG)
        second_cost = np.minimum(second_cost, candidate)
    unique = best_cost * (1 + uniqueness) <= second_cost

    # ---------- 6. left-right consistency check ----------
    # Do the matching from the other side: a pixel at x in the RIGHT image matches the
    # pixel at x + d in the LEFT image, so its cost is total[y, x + d, i].
    report(0.85, "Left-right check")
    total_r = np.full_like(total, BIG)
    for i in range(D):
        d = shifts[i]
        if d >= 0:
            total_r[:, :W - d, i] = total[:, d:, i]
        else:
            total_r[:, -d:, i] = total[:, :W + d, i]
    best_r = total_r.argmin(axis=2) + min_d  # best shift (in pixels) as seen from the right image
    del total, total_r

    d_left = best + min_d                    # best shift (in pixels) as seen from the left image

    # For every pixel, look up what the right image says about the matching pixel.
    # We do it one shift value at a time: pixels whose shift is d are looked up at x - d.
    columns = np.tile(np.arange(W), (H, 1))  # columns[y, x] = x
    d_back = np.zeros((H, W), dtype=np.int64)
    inside = np.zeros((H, W), dtype=bool)
    for d in shifts:
        this_d = (d_left == d)                               # pixels whose best shift is d
        best_r_moved = shift_columns(best_r, d)              # best_r_moved[y, x] = best_r[y, x - d]
        d_back[this_d] = best_r_moved[this_d]
        x_in_right = columns - d                             # column of the matching pixel in the right image
        inside[this_d] = (x_in_right[this_d] >= 0) & (x_in_right[this_d] < W)

    consistent = np.abs(d_left - d_back) <= 1                # both sides agree (within 1 pixel)
    valid = unique & consistent & inside

    # ---------- 7. fill holes, clean up, disparity -> elevation ----------
    report(0.93, "Filling holes and smoothing")
    disparity = fill_holes(disparity, valid)
    disparity = median_filter(disparity, 5)

    low, high = np.percentile(disparity, [0.5, 99.5])        # ignore the most extreme 1% of values
    disparity = np.clip(disparity, low, high)

    # more shift = closer to the camera = higher ground
    elevation = gaussian_blur(disparity - disparity.min())
    report(1.0, "Done")

    return {
        "left_rgb": left_rgb,
        "right_rgb": right_rgb,
        "disparity": disparity.astype(np.float32),
        "elevation": np.asarray(elevation, dtype=np.float32),
        "valid_fraction": float(valid.mean()),
        "seconds": time.time() - t0,
    }


# =====================================================================
#  Command-line mode (matplotlib)
# =====================================================================
def plot_matplotlib(res, out_png="dem_scratch_output.png", show=True):
    import matplotlib.pyplot as plt

    left_rgb, right_rgb = res["left_rgb"], res["right_rgb"]
    disparity, elevation = res["disparity"], res["elevation"]
    H, W = elevation.shape

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

    # 3D surface: only use every `step`-th pixel so it draws fast
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
    plt.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    if show:
        plt.show()


def main():
    if len(sys.argv) > 2:
        left_path, right_path = sys.argv[1], sys.argv[2]
    else:
        # in this order craters come out as dips (the other order turns them into hills)
        left_path, right_path = "images/moon1_right.tif", "images/moon1_left.tif"
    min_d = int(sys.argv[3]) if len(sys.argv) > 4 else -8
    max_d = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    print(f"Searching shifts {min_d}..{max_d}")
    res = compute_dem(left_path, right_path, min_d, max_d,
                      progress=lambda f, m: print(f"  {f * 100:3.0f}%  {m}"))
    print(f"Done in {res['seconds']:.1f} s ({res['valid_fraction'] * 100:.0f}% pixels kept)")
    plot_matplotlib(res)


if __name__ == "__main__":
    main()