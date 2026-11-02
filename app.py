"""
Local web UI for dem.py.   Run with:   streamlit run app.py
Upload a stereo pair -> see both images, the disparity heatmap and a rotatable 3D DEM.
"""
import io

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from dem import BLOCK, MAX_PIXELS, P1, P2, compute_dem, estimate_memory_gb

st.set_page_config(page_title="DEM from stereo images", layout="wide")
st.title("DEM from 2 stereo images")

COLORSCALES = ["Plasma", "Viridis", "Earth", "Cividis", "Turbo", "Inferno"]
FILE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]


# ---------------- plotting helpers ----------------
def image_figure(rgb, title):
    fig = px.imshow(rgb, binary_string=True)
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=380)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def heatmap_figure(z, title, colorscale, colorbar_title):
    fig = px.imshow(z, color_continuous_scale=colorscale, aspect="equal")
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=380,
                      coloraxis_colorbar=dict(title=colorbar_title))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def surface_figure(elev, max_points, exaggeration, colorscale, contours):
    step = max(1, int(np.ceil((elev.size / max_points) ** 0.5)))
    z = elev[::step, ::step]
    sh, sw = z.shape
    x = np.arange(sw) * step
    y = np.arange(sh) * step
    surf = go.Surface(x=x, y=y, z=z, colorscale=colorscale,
                      colorbar=dict(title="rel. elevation"))
    if contours:
        surf.update(contours_z=dict(show=True, usecolormap=True, project_z=True))
    longest = max(sw, sh)
    fig = go.Figure(surf)
    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=sw / longest, y=sh / longest, z=0.4 * exaggeration),
            yaxis=dict(autorange="reversed"),   # same orientation as the images
            xaxis_title="x (px)", yaxis_title="y (px)", zaxis_title="elevation",
            camera=dict(eye=dict(x=1.3, y=1.6, z=1.1)),
        ),
    )
    return fig


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Input")
    left_file = st.file_uploader("Left image", type=FILE_TYPES, key="left")
    right_file = st.file_uploader("Right image", type=FILE_TYPES, key="right")
    swap = st.checkbox("Swap left / right", value=False,
                       help="Swapping flips the relief: craters become hills and vice versa.")

    st.header("Matching")
    min_d, max_d = st.slider("Shift range (px)", -64, 128, (-8, 8),
                             help="Search range for the horizontal shift between the two images.")
    max_mpix = st.slider("Max size (megapixels)", 0.1, 3.0, MAX_PIXELS / 1e6, 0.1,
                         help="Bigger images are shrunk to this. Lower = faster, less memory.")
    with st.expander("Advanced"):
        block = st.select_slider("Block size", options=[3, 5, 7, 9, 11, 15], value=BLOCK)
        p1 = st.number_input("P1 (1-px change penalty)", value=float(P1), min_value=0.0)
        p2 = st.number_input("P2 (big jump penalty)", value=float(P2), min_value=0.0)

    run = st.button("Compute DEM", type="primary", disabled=not (left_file and right_file),
                    use_container_width=True)

    # warn about memory before the user clicks
    if left_file:
        try:
            w, h = Image.open(io.BytesIO(left_file.getvalue())).size
            gb = estimate_memory_gb(w, h, max_d - min_d + 1, int(max_mpix * 1e6))
            msg = f"Estimated peak memory: ~{gb:.1f} GB"
            (st.warning if gb > 4 else st.caption)(msg)
        except Exception:
            pass

# ---------------- compute ----------------
if run:
    lb, rb = left_file.getvalue(), right_file.getvalue()
    if swap:
        lb, rb = rb, lb
    bar = st.progress(0.0, text="Starting")
    try:
        result = compute_dem(
            io.BytesIO(lb), io.BytesIO(rb), min_d, max_d,
            max_pixels=int(max_mpix * 1e6), block=block, p1=p1, p2=p2,
            progress=lambda f, m: bar.progress(min(f, 1.0), text=m),
        )
        st.session_state["result"] = result
    except Exception as e:  # bad size, too small range, out of memory, ...
        st.session_state.pop("result", None)
        st.error(f"Could not compute DEM: {e}")
    finally:
        bar.empty()

result = st.session_state.get("result")
if result is None:
    st.info("Upload a left and a right image in the sidebar, then press **Compute DEM**.")
    st.stop()

# ---------------- display options (no recompute needed) ----------------
with st.sidebar:
    st.header("Display")
    colorscale = st.selectbox("Colour scale", COLORSCALES)
    invert = st.checkbox("Invert elevation", value=False)
    exaggeration = st.slider("Vertical exaggeration", 0.2, 5.0, 1.0, 0.1)
    max_points = st.select_slider("3D detail (points)", options=[10_000, 20_000, 50_000, 100_000, 200_000],
                                  value=50_000)
    contours = st.checkbox("Contour lines on 3D surface", value=True)

elevation = result["elevation"]
if invert:
    elevation = elevation.max() - elevation

st.caption(f"Computed in {result['seconds']:.1f} s | "
           f"{result['valid_fraction'] * 100:.0f}% of pixels passed the consistency checks | "
           f"{elevation.shape[1]} x {elevation.shape[0]} px")

c1, c2, c3 = st.columns(3)
c1.plotly_chart(image_figure(result["left_rgb"], "Image 1 (left)"))
c2.plotly_chart(image_figure(result["right_rgb"], "Image 2 (right)"))
c3.plotly_chart(heatmap_figure(result["disparity"], "Disparity heatmap", colorscale, "px"))

st.subheader("DEM - 3D surface (drag to rotate, scroll to zoom)")
st.plotly_chart(surface_figure(elevation, max_points, exaggeration, colorscale, contours))

buf = io.BytesIO()
np.save(buf, elevation)
st.download_button("Download DEM (.npy)", buf.getvalue(), file_name="dem.npy")