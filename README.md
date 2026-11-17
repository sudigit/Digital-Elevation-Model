# Digital Elevation Model (DEM) from Stereo Images

An end-to-end stereo reconstruction pipeline implemented from scratch in **pure NumPy** (no OpenCV or heavy computer vision framework dependencies). 

This project computes a 3D Digital Elevation Model (DEM) from a pair of stereo images using Semi-Global Matching (SGM), sub-pixel refinement, and validation filters. It includes both an interactive **Streamlit web application** with rotatable 3D surface rendering (Plotly) and a **command-line script** with Matplotlib visualizations.

---

## ✨ Features

- **Pure NumPy Implementation**: Stereo matching, cost computation, and dynamic programming written from scratch with transparent, readable array operations.
- **Semi-Global Matching (SGM)**: 4-directional path cost aggregation (horizontal and vertical passes) using smoothness penalties ($P_1$ and $P_2$) to handle low-texture regions and maintain sharp depth boundaries.
- **Sub-Pixel Parabolic Refinement**: Fits quadratic curves to discrete matching costs for continuous, smooth elevation transitions.
- **Robust Outlier & Occlusion Handling**:
  - **Uniqueness Check**: Rejects ambiguous matches where the winning shift does not sufficiently beat second-best alternatives.
  - **Left-Right Consistency Check**: Cross-validates disparity bids from both vantage points to filter out occluded regions.
  - **Hole Filling & Spatial Filtering**: Row-wise nearest-neighbor hole filling, spike removal via median filtering, and Gaussian surface smoothing.
- **Interactive Web App (Streamlit + Plotly)**:
  - Drag-and-drop stereo image pair uploader (PNG, JPG, TIF).
  - Pre-computation peak memory estimation.
  - Real-time parameter tuning: shift range, block size, SGM penalties, and downsampling limit.
  - Interactive 3D surface plot with zoom, rotation, contour projections, vertical exaggeration, and custom colormaps.
  - One-click `.npy` elevation array export.
- **Command-Line Interface (CLI)**: Run headless or generate standalone Matplotlib figures with multi-panel views and 3D surface plots.
- **Bundled Sample Dataset**: Includes high-resolution lunar stereo imagery (`images/moon1_left.tif`, `images/moon1_right.tif`) sourced from [NASA SVS #3531](https://svs.gsfc.nasa.gov/3531/).

---

## 📁 Repository Structure

```text
Digital-Elevation-Model/
├── images/
│   ├── moon1_left.tif        # Sample lunar stereo image (left vantage)
│   └── moon1_right.tif       # Sample lunar stereo image (right vantage)
├── app.py                    # Streamlit web application with interactive 3D Plotly view
├── dem.py                    # Core stereo matching pipeline & Matplotlib CLI runner
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🔬 How the Pipeline Works

The algorithm reconstructs elevation from a pair of rectified stereo images in seven sequential stages:

```
[Stereo Pair (L/R)]
        │
        ▼
1. Preprocessing & Normalization ──► Luminance conversion + contrast/mean matching
        │
        ▼
2. Matching Cost Volume (SAD)   ──► Sum of Absolute Differences over k×k windows
        │
        ▼
3. 4-Path SGM Aggregation       ──► Dynamic programming (L→R, R→L, T→B, B→T) with P1/P2
        │
        ▼
4. Winner-Takes-All & Sub-pixel ──► Argmin cost selection + quadratic curve interpolation
        │
        ▼
5. Validation Filters           ──► Uniqueness test + Left-Right consistency check
        │
        ▼
6. Post-Processing              ──► Hole filling + median filter + Gaussian blur
        │
        ▼
7. Elevation Surface            ──► Relative height generation & 3D mesh rendering
```

1. **Preprocessing & Normalization**:
   Images are resized if their resolution exceeds `MAX_PIXELS` to manage memory. They are converted to grayscale using standard luminance weights ($0.299R + 0.587G + 0.114B$) and the right image is normalized to match the mean and standard deviation of the left image.
2. **Matching Cost Computation (SAD)**:
   For every candidate horizontal shift $d \in [d_{\min}, d_{\max}]$, the right image is shifted and the pixel-wise absolute difference with the left image is smoothed with a box filter of size $k \times k$.
3. **Semi-Global Matching (SGM)**:
   Path costs are aggregated along 4 directions (left $\to$ right, right $\to$ left, top $\to$ bottom, bottom $\to$ top). Small 1-pixel shift changes incur a penalty $P_1$, while larger disparity discontinuities incur a penalty $P_2$.
4. **Disparity Selection & Sub-Pixel Refinement**:
   The winner-takes-all (WTA) lowest-cost shift index is selected for each pixel. A 3-point parabolic fit across the winning cost and its immediate neighbors yields fractional pixel disparity offsets.
5. **Consistency Checks & Validation**:
   - *Uniqueness*: Discards pixels where the lowest cost does not beat non-neighboring shifts by at least $(1 + \text{uniqueness})$.
   - *Left-Right Cross-Check*: Computes disparities from the right perspective and flags pixels that disagree by more than 1 pixel.
6. **Hole Filling & Spatial Filtering**:
   Invalid/occluded pixels are filled using the smaller neighboring valid disparity along the row. A $5 \times 5$ median filter removes salt-and-pepper noise, extreme 1% outliers are clipped, and a Gaussian blur smooths high-frequency artifacts.
7. **Elevation Output**:
   Disparity is mapped to relative elevation (greater shift indicates closer proximity to the sensor, corresponding to higher topography).

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- `pip` package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Digital-Elevation-Model.git
   cd Digital-Elevation-Model
   ```

2. **(Optional) Create and activate a virtual environment**:
   - **Windows**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### 1. Interactive Web Application (Streamlit)

Launch the interactive browser UI:

```bash
streamlit run app.py
```

- **Upload Stereo Pairs**: Upload any corresponding left and right images.
- **Relief Swap**: Toggle **Swap left / right** to invert parallax if craters appear as mounds or vice-versa.
- **Adjust Parameters**: Configure disparity search bounds, maximum megapixels, window block size, and penalties ($P_1, P_2$).
- **Inspect Results**:
  - View side-by-side rectified source images and the disparity heatmap.
  - Interact with the 3D surface plot (rotate, pan, zoom).
  - Adjust vertical exaggeration, color scale (`Plasma`, `Viridis`, `Earth`, etc.), and surface contour lines without recomputing.
- **Export Data**: Download the resulting elevation grid directly as a `.npy` file.

### 2. Command-Line Interface (Matplotlib)

Run the stereo pipeline directly from the terminal:

```bash
# Run with bundled lunar sample images (default shift range -8 to 8):
python dem.py

# Run with custom images and a specific shift range:
python dem.py images/moon1_right.tif images/moon1_left.tif -8 8
```

Output:
- Prints progress percentages and execution timing to the terminal.
- Saves a high-resolution visualization figure to `dem_scratch_output.png`.
- Displays an interactive Matplotlib window containing:
  - Input Left & Right images
  - Disparity heatmap
  - Contour elevation map
  - 3D perspective surface projection

---

## ⚙️ Parameters & Tuning Guide

| Parameter | Default | Recommended Range | Description |
|---|---|---|---|
| `min_d`, `max_d` | `-8, 8` | Dataset dependent | Range of horizontal pixel disparity to search. Must cover minimum and maximum physical parallax. |
| `max_pixels` | `1,500,000` | `500,000 - 3,000,000` | Downscaling threshold to prevent excessive memory and compute times on large sensor images. |
| `block` | `7` | `3, 5, 7, 9, 11` | Matching window size (must be odd). Larger values handle low texture better but round off sharp boundaries. |
| `p1` | `4.0` | `2.0 - 10.0` | SGM penalty for small (1 px) disparity changes. Encourages smooth slopes. |
| `p2` | `48.0` | `20.0 - 100.0` | SGM penalty for large (>1 px) disparity jumps. Must satisfy $P_2 > P_1$. |
| `uniqueness` | `0.05` | `0.02 - 0.15` | Minimum relative margin required for winning cost over secondary minima. |
| `Vertical Exaggeration` | `1.0` | `0.2 - 5.0` | (Web UI only) Visual scaling factor for the z-axis to highlight subtle topographic variations. |

> [!TIP]
> **Relief Inversion**: If relief appears inverted (e.g., impact craters look like domes), swap the left and right image inputs or check the **Swap left / right** checkbox in the Web UI.

---

## 📦 Dependencies

All dependencies are standard and lightweight:

- [`numpy`](https://numpy.org/) — Vectorized array operations and dynamic programming
- [`pillow`](https://python-pillow.org/) — Image I/O and format conversion
- [`streamlit`](https://streamlit.io/) — Interactive web interface
- [`plotly`](https://plotly.com/python/) — Hardware-accelerated 3D surface visualization
- [`matplotlib`](https://matplotlib.org/) — Command-line 2D/3D plotting

---

## 📚 References & Data Sources

### Data Sources
- **Lunar Stereo Imagery**: [NASA's Scientific Visualization Studio (SVS Entry #3531)](https://svs.gsfc.nasa.gov/3531/) — High-resolution lunar surface stereo photography captured by the Apollo metric mapping cameras.

### Algorithms & Literature
- **Semi-Global Matching (SGM)**:
  - [Semi-Global Matching Tutorial](https://engineering.purdue.edu/kak/Tutorials/SemiGlobalMatching.pdf) — Prof. Avi Kak, Purdue University (an accessible tutorial on directional dynamic programming for stereo vision).
  - Hirschmüller, H. (2008). *Stereo Processing by Semiglobal Matching and Mutual Information*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 30(2), 328–341.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

