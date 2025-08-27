import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path
from collections import deque

import PySpin  # Spinnaker/PySpin

# =========================
# Live Laser-Line Monitor (PySpin)
# =========================

# ---- PySpin camera index ----
CAM_INDEX = 0  # pick your camera if multiple are connected

# ---- Processing Tunables ----
MIN_INTENSITY   = 70     # reject weak columns
SMOOTH_K        = 10     # median filter length on y(x) (odd)
MAD_K           = 2.9    # threshold = MAD_K * MAD
ABS_MIN_DEV_PX  = 0.6    # floor on threshold (px)
MIN_EVENT_WIDTH = 12     # ignore tiny spikes (px)
DOWNSCALE_W     = 1024   # process width for speed (None = full width)


# ---- ROI (reduce work & reject noise outside the band) ----
# Set to None to use whole frame, or set a vertical slice (top, bottom) in pixels.
ROI_Y = None  # e.g., ROI_Y = (200, 500)

# ---- Display ----
PLOT_H = 240
PLOT_W = 1024
FONT   = cv2.FONT_HERSHEY_SIMPLEX

# ---- Output ----
OUT_DIR = Path("live_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def postprocess_and_save(img_bgr,
                         out_png=OUT_DIR / "laser_line_anomalies.png",
                         out_csv_events=OUT_DIR / "laser_line_anomalies.csv",
                         out_csv_profile=OUT_DIR / "laser_line_profile.csv"):
    """
    Run the anomaly detector on the CURRENT frame and save one overlay PNG (and CSVs).
    Returns overlay_bgr, events (list of dicts).
    """
    h, w = img_bgr.shape[:2]

    # Optional ROI + downscale (reuse your globals)
    y0, y1 = (0, h) if ROI_Y is None else ROI_Y
    y0 = max(0, y0); y1 = min(h, y1)
    roi = img_bgr[y0:y1]
    proc = roi
    scale = 1.0
    if DOWNSCALE_W is not None and proc.shape[1] > DOWNSCALE_W:
        scale = DOWNSCALE_W / proc.shape[1]
        proc = cv2.resize(proc, (DOWNSCALE_W, int(proc.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    ph, pw = proc.shape[:2]

    score = isolate_laser_score(proc)
    norm = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bg   = cv2.morphologyEx(norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)))
    enh  = cv2.subtract(norm, bg)
    enh_blur = cv2.GaussianBlur(enh, (3,3), 0)

    # subpixel ridge per column
    y_subpix = np.full(pw, np.nan, dtype=np.float32)
    for x in range(pw):
        col = enh_blur[:, x].astype(np.float32)
        y0c = int(np.argmax(col))
        s0  = col[y0c]
        if s0 < MIN_INTENSITY:
            continue
        if 1 <= y0c < ph-1:
            d = subpixel_peak_quadratic(col[y0c-1], col[y0c], col[y0c+1])
            y_sp = y0c + d
        else:
            y_sp = float(y0c)
        y_subpix[x] = y_sp

    valid = ~np.isnan(y_subpix)
    xs = np.arange(pw)[valid]
    if xs.size == 0:
        # Save a note image so it's obvious
        overlay = img_bgr.copy()
        cv2.putText(overlay, "No laser line detected", (20, 40), FONT, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_png), overlay)
        return overlay, []

    ys = y_subpix[valid].astype(np.float32)

    # smooth (median)
    ys = median_filter_1d(ys, k=SMOOTH_K if SMOOTH_K % 2 == 1 else SMOOTH_K+1)

    # detrend to straight line
    m, b = np.polyfit(xs, ys, 1)
    baseline = m*xs + b
    residual = ys - baseline

    # robust threshold
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-9
    thresh = max(MAD_K * 1.4826 * mad, ABS_MIN_DEV_PX)

    # contiguous segments
    mask = np.abs(residual - med) >= thresh
    events = []
    if mask.any():
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

    # Build overlay in full-res coords, draw ONLY the line and anomalies
    overlay = img_bgr.copy()
    xs_full = (xs / scale).astype(int)
    ys_full = (ys / scale + y0).astype(float)

    # thin green line
    for x_i, y_i in zip(xs_full, ys_full):
        if 0 <= x_i < w:
            yy = int(round(y_i))
            if 0 <= yy < h:
                cv2.circle(overlay, (int(x_i), yy), 1, (0,255,0), -1)

    # red thicker dots for anomaly spans
    for e in events:
        x0p = int(e["start_x"] / scale)
        x1p = int(e["end_x"]   / scale)
        for xi in range(x0p, x1p+1):
            idx = np.clip(int(round(xi*scale)), 0, len(xs)-1)
            yv = ys_full[idx]
            yy = int(round(yv))
            if 0 <= xi < w and 0 <= yy < h:
                cv2.circle(overlay, (int(xi), yy), 2, (0,0,255), -1)

    # Save outputs
    pd.DataFrame(events, columns=["start_x","end_x","width_px","peak_x","peak_dev_px","mean_dev_px"]).to_csv(out_csv_events, index=False)
    prof = np.stack([xs, ys, baseline, residual], axis=1)
    pd.DataFrame(prof, columns=["x","y","baseline","residual"]).to_csv(out_csv_profile, index=False)
    cv2.imwrite(str(out_png), overlay)

    return overlay, events


# ----------------- Helpers (from your batch script) -----------------
def subpixel_peak_quadratic(fm1, f0, fp1):
    denom = (fm1 - 2.0 * f0 + fp1)
    if denom == 0:
        return 0.0
    return 0.5 * (fm1 - fp1) / denom

def isolate_laser_score(img):
    """
    Returns a per-pixel 'laser-ness' score.
    - If 3-channel: use chroma (R or G minus average of the others).
    - If 1-channel: just use the intensity.
    """
    if img.ndim == 2:
        score = img.astype(np.float32)
    elif img.shape[2] == 1:
        score = img[:, :, 0].astype(np.float32)
    else:
        b, g, r = cv2.split(img)
        score_r = r.astype(np.float32) - 0.5 * (g.astype(np.float32) + b.astype(np.float32))
        score_g = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
        score = np.maximum(score_r, score_g)
    score = np.clip(score, 0, None)
    score = cv2.GaussianBlur(score, (5, 5), 0)
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

def draw_residual_plot(xs, residual, h=PLOT_H, w=PLOT_W, margin=30, thresh=None):
    plot = np.full((h, w, 3), 255, np.uint8)
    cv2.rectangle(plot, (0,0), (w-1,h-1), (220,220,220), 1)
    if xs.size == 0:
        cv2.putText(plot, "No data", (20, h//2), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)
        return plot

    x_norm = (xs - xs.min()) / max(1, (xs.max() - xs.min()))
    x_pix = (margin + x_norm*(w-2*margin)).astype(int)

    mid = h//2
    r_abs_max = max(ABS_MIN_DEV_PX, float(np.max(np.abs(residual)))) if residual.size else ABS_MIN_DEV_PX
    pixels_per_unit = 0.9*(h//2 - margin) / r_abs_max

    cv2.line(plot, (margin, mid), (w-margin, mid), (0,0,0), 1)
    for k in [-2,-1,1,2]:
        y = int(mid - k * ABS_MIN_DEV_PX * pixels_per_unit)
        cv2.line(plot, (margin, y), (w-margin, y), (230,230,230), 1)

    pts = []
    for xi, r in zip(x_pix, residual):
        yi = int(round(mid - r * pixels_per_unit))
        pts.append((int(xi), int(np.clip(yi, 0, h-1))))
    if len(pts) >= 2:
        cv2.polylines(plot, [np.array(pts, dtype=np.int32)], False, (40,40,40), 1, cv2.LINE_AA)

    if thresh is not None:
        y_up = int(round(mid - thresh * pixels_per_unit))
        y_dn = int(round(mid + thresh * pixels_per_unit))
        cv2.line(plot, (margin, y_up), (w-margin, y_up), (0,0,255), 1, cv2.LINE_AA)
        cv2.line(plot, (margin, y_dn), (w-margin, y_dn), (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(plot, f"+/- {thresh:.3f}px", (margin+5, y_up-6), FONT, 0.5, (0,0,255), 1, cv2.LINE_AA)

    cv2.putText(plot, "Residual vs x (px)", (margin, 22), FONT, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return plot

def find_anomalies_from_frame(img_bgr):
    """Returns overlay_bgr, xs, residual, events, baseline, ys."""
    h, w = img_bgr.shape[:2]

    y0, y1 = (0, h) if ROI_Y is None else ROI_Y
    y0 = max(0, y0); y1 = min(h, y1)
    roi = img_bgr[y0:y1]

    proc = roi
    scale = 1.0
    if DOWNSCALE_W is not None and proc.shape[1] > DOWNSCALE_W:
        scale = DOWNSCALE_W / proc.shape[1]
        proc = cv2.resize(proc, (DOWNSCALE_W, int(proc.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    ph, pw = proc.shape[:2]

    score = isolate_laser_score(proc)
    norm  = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    bg    = cv2.morphologyEx(norm, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)))
    enh   = cv2.subtract(norm, bg)
    enh_blur = cv2.GaussianBlur(enh, (3,3), 0)

    y_subpix = np.full(pw, np.nan, dtype=np.float32)
    for x in range(pw):
        col = enh_blur[:, x].astype(np.float32)
        y0c = int(np.argmax(col))
        s0 = col[y0c]
        if s0 < MIN_INTENSITY:
            continue
        if 1 <= y0c < ph-1:
            d = subpixel_peak_quadratic(col[y0c-1], col[y0c], col[y0c+1])
            y_sp = y0c + d
        else:
            y_sp = float(y0c)
        y_subpix[x] = y_sp

    valid = ~np.isnan(y_subpix)
    xs = np.arange(pw)[valid]
    if xs.size == 0:
        overlay = img_bgr.copy()
        cv2.putText(overlay, "No laser line detected", (20, 40), FONT, 1, (0,0,255), 2, cv2.LINE_AA)
        return overlay, np.array([]), np.array([]), [], np.array([]), np.array([])

    ys = y_subpix[valid]
    ys = median_filter_1d(ys.astype(np.float32), k=SMOOTH_K)

    m, b = np.polyfit(xs, ys, 1)
    baseline = m*xs + b
    residual = ys - baseline

    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-9
    thresh = max(MAD_K * 1.4826 * mad, ABS_MIN_DEV_PX)

    mask = np.abs(residual - med) >= thresh
    events = []
    if mask.any():
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

    overlay = img_bgr.copy()
    if ROI_Y is not None:
        cv2.rectangle(overlay, (0, y0), (w, y1-1), (255, 255, 0), 1)

    xs_full = (xs / scale).astype(int)
    ys_full = (ys / scale + y0).astype(float)

    for x_i, y_i in zip(xs_full, ys_full):
        if 0 <= x_i < w and 0 <= int(round(y_i)) < h:
            cv2.circle(overlay, (int(x_i), int(round(y_i))), 1, (0,255,0), -1)

    for e in events:
        x0p = int(e["start_x"] / scale)
        x1p = int(e["end_x"]   / scale)
        for xi in range(x0p, x1p+1, 1):
            idx = np.clip(int(round(xi*scale)), 0, len(xs)-1)
            yv = ys_full[idx]
            if 0 <= xi < w and 0 <= int(round(yv)) < h:
                cv2.circle(overlay, (int(xi), int(round(yv))), 2, (0,0,255), -1)

    cv2.putText(overlay, f"thresh={thresh:.3f}px  MAD={mad:.3f}px  floor={ABS_MIN_DEV_PX}",
                (20, 30), FONT, 0.6, (0,255,255), 2, cv2.LINE_AA)

    return overlay, xs, residual, events, baseline, ys

# -------------------- PySpin setup helpers --------------------
def set_node_safely(node_map, name, value):
    """Attempt to set a GenICam node if it exists and is writable."""
    try:
        node = PySpin.CEnumerationPtr(node_map.GetNode(name))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            return False
        # value can be a string enum entry or integer
        if isinstance(value, str):
            entry = node.GetEntryByName(value)
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                node.SetIntValue(entry.GetValue())
                return True
        else:
            node.SetIntValue(int(value))
            return True
    except Exception:
        pass
    return False

def set_float_node(node_map, name, value):
    try:
        node = PySpin.CFloatPtr(node_map.GetNode(name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            mn, mx = node.GetMin(), node.GetMax()
            node.SetValue(float(np.clip(value, mn, mx)))
            return True
    except Exception:
        pass
    return False

# def configure_camera(cam):
#     """Optional: tweak common settings (pixel format, exposure, gain, FPS)."""
#     sNodemap = cam.GetTLStreamNodeMap()
#     # Recommended: enable stream buffer handling "NewestOnly" to reduce latency
#     try:
#         handling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
#         if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
#             newest_only = handling_mode.GetEntryByName('NewestOnly')
#             if PySpin.IsAvailable(newest_only) and PySpin.IsReadable(newest_only):
#                 handling_mode.SetIntValue(newest_only.GetValue())
#     except Exception:
#         pass

#     nodemap = cam.GetNodeMap()

#     # Turn off auto exposure/gain if you want stable intensity
#     set_node_safely(nodemap, 'ExposureAuto', 'Off')
#     set_float_node(nodemap, 'ExposureTime', 8000.0)  # µs, adjust to your laser brightness
#     set_node_safely(nodemap, 'GainAuto', 'Off')
#     set_float_node(nodemap, 'Gain', 0.0)

#     # Frame rate (some models require disabling exposure auto first)
#     set_node_safely(nodemap, 'AcquisitionFrameRateEnable', 1)
#     set_float_node(nodemap, 'AcquisitionFrameRate', 60.0)  # target FPS

#     # Pixel format: prefer BGR8 or RGB8 for simple conversion
#     # If camera doesn’t offer BGR8 directly, we’ll Convert() later.
#     set_node_safely(nodemap, 'PixelFormat', 'BayerRG8') or \
#     set_node_safely(nodemap, 'PixelFormat', 'Mono8') or \
#     set_node_safely(nodemap, 'PixelFormat', 'RGB8')
def configure_camera(cam):
    """
    Configure camera settings to match SpinView screenshot + BGR8 output.
    """
    nodemap = cam.GetNodeMap()
    sNodemap = cam.GetTLStreamNodeMap()

    # --- Buffer handling ---
    try:
        handling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName('NewestOnly')
            if PySpin.IsAvailable(newest_only) and PySpin.IsReadable(newest_only):
                handling_mode.SetIntValue(newest_only.GetValue())
    except Exception:
        pass

    # --- Acquisition mode ---
    set_node_safely(nodemap, 'AcquisitionMode', 'Continuous')

    # --- Frame rate ---
    set_node_safely(nodemap, 'AcquisitionFrameRateEnable', 1)
    set_float_node(nodemap, 'AcquisitionFrameRate', 964.28)   # Hz target

    # --- Exposure ---
    set_node_safely(nodemap, 'ExposureAuto', 'Off')
    set_node_safely(nodemap, 'ExposureMode', 'Timed')
    set_float_node(nodemap, 'ExposureTime', 503.0)            # µs
    # optional: set lower/upper limits
    # set_float_node(nodemap, 'ExposureTimeLowerLimit', 100.0)
    # set_float_node(nodemap, 'ExposureTimeUpperLimit', 15000.0)

    # --- Gain ---
    set_node_safely(nodemap, 'GainAuto', 'Off')
    set_float_node(nodemap, 'Gain', 6.9)                      # dB

    # --- Gamma ---
    set_node_safely(nodemap, 'GammaEnable', 1)
    set_float_node(nodemap, 'Gamma', 0.79)

    # --- Black level ---
    set_float_node(nodemap, 'BlackLevel', 0.0)

    # --- White balance ---
    set_node_safely(nodemap, 'BalanceWhiteAuto', 'Off')
    set_node_safely(nodemap, 'BalanceRatioSelector', 'Red')
    set_float_node(nodemap, 'BalanceRatio', 1.48)

    # --- Pixel format (force BGR8 output) ---
    set_node_safely(nodemap, 'PixelFormat', 'BGR8') or \
    set_node_safely(nodemap, 'PixelFormat', 'RGB8') or \
    set_node_safely(nodemap, 'PixelFormat', 'RGB8')


# -------------------- Main loop --------------------
def main():
    system = None
    cam_list = None
    cam = None
    try:
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            raise RuntimeError("No FLIR/Spinnaker cameras found.")

        cam = cam_list.GetByIndex(CAM_INDEX)
        cam.Init()
        configure_camera(cam)
        cam.BeginAcquisition()
        # proc = PySpin.ImageProcessor()
        # proc.SetColorProcessing(PySpin.ColorProcessingAlgorithm_HQ_LINEAR) 

        # cv2.namedWindow("Laser Overlay", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Residual Plot", cv2.WINDOW_NORMAL)

        AUTO_SAVE_ON_FIRST_FRAME = False  # set True to save one PNG then exit immediately

        cv2.namedWindow("Laser Overlay", cv2.WINDOW_NORMAL)

        saved_once = False
        while True:
            # t0 = time.time()
            img = cam.GetNextImage()
            if img.IsIncomplete():
                img.Release()
                continue

            try:
                # Always convert to BGR8 (works from Mono/Bayer)
                # bgr_img = img.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                frame = img.GetNDArray().copy()
            except Exception:
                arr = img.GetNDArray()
                if arr.ndim == 2:
                    frame = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                else:
                    frame = arr[..., ::-1].copy() if arr.shape[-1] == 3 else arr
            finally:
                img.Release()
            image = frame.copy()
            # (Optional) quick FPS overlay for live view
            # dt = time.time() - t0
            # fps = (1.0/dt) if dt > 0 else 0.0
            # cv2.putText(frame, f"{fps:5.1f} FPS", (20, 30), FONT, 0.8, (0,255,0), 2, cv2.LINE_AA)

            # Show the raw or last overlay? For clarity show the live raw feed.
            cv2.imshow("Laser Overlay", frame)

            if AUTO_SAVE_ON_FIRST_FRAME and not saved_once:
                overlay, events = postprocess_and_save(image)
                cv2.imshow("Laser Overlay", overlay)  # replace the view with the saved overlay
                saved_once = True
                break  # exit after saving

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                # run post-processing on the CURRENT frame and save PNG/CSVs
                overlay, events = postprocess_and_save(image)
                cv2.imshow("Laser Overlay", overlay)
                print(f"Saved overlay to: {OUT_DIR / 'laser_line_anomalies.png'}  (events: {len(events)})")

    finally:
        # Graceful cleanup (mirrors your example’s pattern)
        try:
            if cam is not None:
                cam.EndAcquisition()
        except Exception:
            pass
        try:
            if cam is not None:
                cam.DeInit()
        except Exception:
            pass
        try:
            if cam_list is not None:
                cam_list.Clear()
        except Exception:
            pass
        try:
            if cam is not None:
                del cam
            if cam_list is not None:
                del cam_list
        except Exception:
            pass
        try:
            if system is not None:
                system.ReleaseInstance()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
