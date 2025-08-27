# Laser line anomaly detection (small curves along a line)
# - Detrend to a straight line
# - Compute robust residuals
# - Threshold with MAD
# - Group contiguous pixels into events
# - Save CSV and visual overlays

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# from caas_jupyter_tools import display_dataframe_to_user
# 22157381_20250812150806.bmp -- good
# 22157381_20250812131251.bmp -- bad
# 22157381_20250812150930.bmp -- bad
# 22157381_20250812162308.bmp
# 22157381_20250812162040.bmp
# 22157381_20250812162437.bmp
# 22157381_20250812162308.bmp
# 22157381_20250812162605.bmp
# 22157381_20250812162659.bmp
# 22157381_20250814132328.bmp

# 22157381_20250814133439.bmp -- bad
# 22157381_20250814134334.bmp -- good
# 22157381_20250814134634.bmp -- good
# 22157381_20250814140119.bmp -- bad
# 22157381_20250814140209.bmp -- bad
# 22157381_20250814140302.bmp -- good
# 22157381_20250814140342.bmp -- bad
# 22157381_20250814140418.bmp -- good 
# 22157381_20250814140449.bmp -- bad  
# 22157381_20250814140526.bmp -- bad 
# 22157381_20250814140557.bmp -- good
# 22157381_20250814151209.bmp -- bad

# 22157381_20250815204124
# 22157381_20250815204015
# 22157381_20250815203915
# 22157381_20250815205052.bmp
IMG_PATH = "line_scan_data/22157381_20250815205052.bmp"

# ---- Tunables ----
# MIN_INTENSITY = 10          # reject weak columns
# SMOOTH_K = 9                # median filter on y(x) to suppress noise (odd)
# MAD_K = 2                # threshold = MAD_K * MAD
# ABS_MIN_DEV_PX = 0.1        # floor on threshold (px)
# MIN_EVENT_WIDTH = 3         # ignore tiny spikes (px)
# ------------------

MIN_INTENSITY = 80          # reject weak columns
SMOOTH_K = 10                # median filter on y(x) to suppress noise (odd)
MAD_K = 2                # threshold = MAD_K * MAD
ABS_MIN_DEV_PX = 0.4        # floor on threshold (px)
MIN_EVENT_WIDTH = 10        # ignore tiny spikes (px)

out_csv_profile = "laser_line_profile.csv"        # reused if exists
out_csv_events  = "laser_line_anomalies.csv"
out_overlay     = "laser_line_anomalies.png"

def subpixel_peak_quadratic(fm1, f0, fp1):
    denom = (fm1 - 2.0 * f0 + fp1)
    if denom == 0:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom

def isolate_laser_score(bgr):
    b, g, r = cv2.split(bgr)
    score_r = r.astype(np.float32) - 0.5 * (g.astype(np.float32) + b.astype(np.float32))
    score_g = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
    score = np.maximum(score_r, score_g)
    score = np.clip(score, 0, None)
    score = cv2.GaussianBlur(score, (5,5), 0)
    return score

def median_filter_1d(a, k=9):
    if k < 3 or k % 2 == 0:
        return a.copy()
    pad = k // 2
    ap = np.pad(a, (pad, pad), mode='edge')
    out = np.empty_like(a)
    for i in range(len(a)):
        out[i] = np.median(ap[i:i+k])
    return out

# ---- Extract subpixel line ----
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(IMG_PATH)

h, w = img.shape[:2]
score = isolate_laser_score(img)
norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
bg = cv2.morphologyEx(norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)))
enh = cv2.subtract(norm, bg)
enh_blur = cv2.GaussianBlur(enh, (3,3), 0)

y_subpix = np.full(w, np.nan, dtype=np.float32)
strength = np.zeros(w, dtype=np.float32)
for x in range(w):
    col = enh_blur[:, x].astype(np.float32)
    y0 = int(np.argmax(col))
    s0 = col[y0]
    if s0 < MIN_INTENSITY:
        continue
    if 1 <= y0 < h-1:
        d = subpixel_peak_quadratic(col[y0-1], col[y0], col[y0+1])
        y_sp = y0 + d
    else:
        y_sp = float(y0)
    y_subpix[x] = y_sp
    strength[x] = s0

valid = ~np.isnan(y_subpix)
xs = np.arange(w)[valid]
ys = y_subpix[valid]

if ys.size == 0:
    raise RuntimeError("No laser line detected.")

ys = median_filter_1d(ys.astype(np.float32), k=SMOOTH_K)

# ---- Detrend to the straight line (global best-fit) ----
m, b = np.polyfit(xs, ys, 1)
baseline = m*xs + b
residual = ys - baseline

# ---- Robust threshold via MAD ----
med = np.median(residual)
mad = np.median(np.abs(residual - med)) + 1e-9
thresh = max(MAD_K * 1.4826 * mad, ABS_MIN_DEV_PX)   # 1.4826 ≈ std from MAD for Gaussian

# ---- Find contiguous anomaly segments ----
mask = np.abs(residual - med) >= thresh
events = []
if mask.any():
    # run-length grouping
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            end = i - 1
            if (xs[end] - xs[start] + 1) >= MIN_EVENT_WIDTH:
                seg_x = xs[start:end+1]
                seg_r = residual[start:end+1]
                peak_idx = np.argmax(np.abs(seg_r))
                events.append({
                    "start_x": float(seg_x[0]),
                    "end_x": float(seg_x[-1]),
                    "width_px": float(seg_x[-1] - seg_x[0] + 1),
                    "peak_x": float(seg_x[peak_idx]),
                    "peak_dev_px": float(seg_r[peak_idx]),
                    "mean_dev_px": float(np.mean(seg_r))
                })
            start = None
    if start is not None:
        end = len(mask) - 1
        if (xs[end] - xs[start] + 1) >= MIN_EVENT_WIDTH:
            seg_x = xs[start:end+1]
            seg_r = residual[start:end+1]
            peak_idx = np.argmax(np.abs(seg_r))
            events.append({
                "start_x": float(seg_x[0]),
                "end_x": float(seg_x[-1]),
                "width_px": float(seg_x[-1] - seg_x[0] + 1),
                "peak_x": float(seg_x[peak_idx]),
                "peak_dev_px": float(seg_r[peak_idx]),
                "mean_dev_px": float(np.mean(seg_r))
            })

events_df = pd.DataFrame(events, columns=["start_x","end_x","width_px","peak_x","peak_dev_px","mean_dev_px"])
events_df.to_csv(out_csv_events, index=False)

# ---- Visuals ----
# 1) Residual plot with threshold
plt.figure()
plt.plot(xs, residual)
plt.axhline(med + thresh)
plt.axhline(med - thresh)
plt.xlabel("x (pixels)")
plt.ylabel("Deviation from straight line (px)")
plt.title("Residuals vs. x (thresholds shown)")
plt.show()

# 2) Overlay on image marking anomalies
overlay = img.copy()
# draw detected subpixel line (thin)
for x_i, y_i in zip(xs.astype(int), ys.astype(float)):
    cv2.circle(overlay, (int(x_i), int(round(y_i))), 1, (0,255,0), -1)
# draw anomaly spans
for e in events:
    x0, x1 = int(e["start_x"]), int(e["end_x"])
    # highlight the span with a thicker polyline at the measured y
    for xi in range(x0, x1+1):
        # find closest index in xs
        # since xs is increasing and contiguous for most images, we can map via searchsorted
        idx = np.searchsorted(xs, xi)
        if idx >= len(xs):
            idx = len(xs)-1
        if xs[idx] != xi and idx > 0 and (xi - xs[idx-1]) < (xs[idx] - xi):
            idx = idx-1
        yv = ys[idx]
        cv2.circle(overlay, (int(xi), int(round(yv))), 2, (0,0,255), -1)  # red markers for anomalies

cv2.imwrite(out_overlay, overlay)

# 3) Show anomalies table preview
# display_dataframe_to_user("Laser line anomalies", events_df)

# Also persist the profile if not already saved by a previous step
csv_arr = np.stack([xs, ys, baseline, residual], axis=1)
pd.DataFrame(csv_arr, columns=["x","y","baseline","residual"]).to_csv(out_csv_profile, index=False)

print(f"Saved anomalies CSV to: {out_csv_events}")
print(f"Saved overlay to: {out_overlay}")
