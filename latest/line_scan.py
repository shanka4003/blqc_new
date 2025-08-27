import cv2
import numpy as np
import pytesseract
import re
import json
import os
import logging
from pyzbar import pyzbar
import time
import math


# # Configure logging for better debugging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def choose_reference_barcode(barcodes):
    """Pick a reference barcode. Here we choose the largest by area."""
    if not barcodes:
        return None, -1
    areas = [(b["position"]["w"] * b["position"]["h"], idx) for idx, b in enumerate(barcodes)]
    _, idx = max(areas)
    return barcodes[idx], idx

def to_relative_box(abs_box, ref_box):
    """
    abs_box: dict {x,y,w,h} in image coords
    ref_box: dict {x,y,w,h} barcode rect in image coords
    Returns normalized offsets relative to barcode (dx,dy,w,h) each / (bw,bh).
    """
    x, y, w, h = abs_box["x"], abs_box["y"], abs_box["w"], abs_box["h"]
    bx, by, bw, bh = ref_box["x"], ref_box["y"], ref_box["w"], ref_box["h"]
    if bw <= 0 or bh <= 0:
        raise ValueError("Invalid reference barcode size for normalization.")
    return {
        "dx": (x - bx) / float(bw),
        "dy": (y - by) / float(bh),
        "w":  w / float(bw),
        "h":  h / float(bh)
    }

def from_relative_box(rel_box, ref_box, clamp_w, clamp_h):
    """
    rel_box: dict {dx,dy,w,h} normalized by barcode size
    ref_box: dict {x,y,w,h} absolute barcode rect
    clamp_w, clamp_h: image width/height for clamping
    Returns absolute pixel box {x,y,w,h} clamped inside image.
    """
    bx, by, bw, bh = ref_box["x"], ref_box["y"], ref_box["w"], ref_box["h"]
    x = int(round(bx + rel_box["dx"] * bw))
    y = int(round(by + rel_box["dy"] * bh))
    w = int(round(rel_box["w"] * bw))
    h = int(round(rel_box["h"] * bh))

    # Clamp
    x = max(0, min(x, clamp_w - 1))
    y = max(0, min(y, clamp_h - 1))
    if x + w > clamp_w: w = max(1, clamp_w - x)
    if y + h > clamp_h: h = max(1, clamp_h - y)
    return {"x": x, "y": y, "w": w, "h": h}

def select_single_roi_and_save(image, key="line_roi", save_path="roi_config.json"):
    """Let user select a single ROI and save it under `key` in roi_config.json."""
    roi = cv2.selectROI(f"Select ROI for {key}", image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi is None or roi == (0, 0, 0, 0):
        logging.warning("No ROI selected. Skipping save.")
        return

    # Load existing config if present
    cfg = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            logging.warning(f"Could not read existing ROI config, starting new. Error: {e}")

    cfg[key] = {"x": int(roi[0]), "y": int(roi[1]), "w": int(roi[2]), "h": int(roi[3])}

    with open(save_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logging.info(f"Saved ROI '{key}' to {save_path}")

import math

def detect_longest_line(
    img,
    roi_config_path="roi_config.json",
    roi_key="line_roi",
    angle_filter_deg=45,      # None => ignore angle; else keep only lines within ±this many degrees of horizontal
    canny_low=40,
    canny_high=150,
    hough_threshold=100,
    hough_min_line_len=10,
    hough_max_line_gap=80,
    draw=True,
    line_color=(255, 178, 0),
    line_thickness=3
):
    try:
        H, W = img.shape[:2]

        # Load ROI if present; else use full image
        roi = {"x": 0, "y": 0, "w": W, "h": H}
        if os.path.exists(roi_config_path):
            try:
                with open(roi_config_path, "r") as f:
                    cfg = json.load(f)
                    if roi_key in cfg:
                        roi = cfg[roi_key]
            except Exception as e:
                logging.warning(f"Failed reading ROI config ({roi_config_path}): {e}")

        x, y, w, h = [int(roi[k]) for k in ["x", "y", "w", "h"]]
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        roi_img = img[y:y+h, x:x+w]
        if roi_img.size == 0:
            logging.warning("Empty ROI for line detection; using full image.")
            roi_img = img
            x, y, w, h = 0, 0, W, H

        # Edges & lines
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180.0,
            threshold=hough_threshold,
            minLineLength=hough_min_line_len,
            maxLineGap=hough_max_line_gap
        )

        best = None  # (length, angle_norm, (x1,y1,x2,y2))
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                dx, dy = (x2 - x1), (y2 - y1)
                length = math.hypot(dx, dy)
                if length <= 0:
                    continue

                angle = math.degrees(math.atan2(dy, dx))          # -180..180
                angle_norm = ((angle + 90) % 180) - 90            # normalize to [-90, 90] so horizontal ~ 0

                # If an angle filter is requested, keep only lines near horizontal
                if angle_filter_deg is not None and abs(angle_norm) > angle_filter_deg:
                    continue

                if (best is None) or (length > best[0]):
                    best = (length, angle_norm, (x1, y1, x2, y2))

        result = {
            "length_px": None,
            "angle_deg": None,                    # signed angle in [-90, 90]
            "tilt_from_horizontal_deg": None,     # |angle_deg|
            "endpoints": None,
            "roi_used": {"x": x, "y": y, "w": w, "h": h}
        }

        if best is not None:
            length, angle_deg, (lx1, ly1, lx2, ly2) = best
            # print(f"Longest line: {length:.1f}px @ {angle_deg:.1f}° (tilt {abs(angle_deg):.1f}°)")
            # Convert to full-image coordinates
            X1, Y1 = lx1 + x, ly1 + y
            X2, Y2 = lx2 + x, ly2 + y

            result.update({
                "length_px": float(length),
                "angle_deg": float(angle_deg),
                "tilt_from_horizontal_deg": float(abs(angle_deg)),
                "endpoints": {"x1": int(X1), "y1": int(Y1), "x2": int(X2), "y2": int(Y2)}
            })

            if draw:
                cv2.line(img, (X1, Y1), (X2, Y2), line_color, line_thickness)
                label = f"{length:.1f}px @ {angle_deg:.3f}° (tilt {abs(angle_deg):.3f}°)"
                tx = min(max(10, min(X1, X2)), W - 260)
                ty = min(max(30, min(Y1, Y2)), H - 10)
                cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        return result

    except Exception as e:
        logging.error(f"Longest-line detection failed: {e}")
        return {
            "length_px": None,
            "angle_deg": None,
            "tilt_from_horizontal_deg": None,
            "endpoints": None,
            "roi_used": None
        }



def enhance_contrast(img):
    """Enhance image contrast using CLAHE on LAB color space."""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(50, 50))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    except Exception as e:
        logging.error(f"Contrast enhancement failed: {e}")
        return img
    
def detect_barcodes(img, roi_top=0.5, roi_bottom=0.9, thresh_range=range(50, 200, 10)):
    """Detect barcodes in the specified ROI of the image."""
    try:
        h = img.shape[0]
        roi = img[int(roi_top * h):int(roi_bottom * h), :]  # Configurable ROI
        if roi.size == 0:
            logging.warning(f"Empty ROI for key. Skipping OCR.")
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        results = []
        best_thresh_image = None

        for thresh_val in thresh_range:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            barcodes = pyzbar.decode(thresh)
            if barcodes:
                for barcode in barcodes:
                    x, y, w, h = barcode.rect
                    # Adjust y-coordinate for full image
                    y_adjusted = y + int(roi_top * img.shape[0])
                    cv2.rectangle(img, (x, y_adjusted), (x + w, y_adjusted + h), (0, 255, 0), 2)
                    text = f"{barcode.type}: {barcode.data.decode()}"
                    cv2.putText(img, text, (x, y_adjusted - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    results.append({
                        "type": barcode.type,
                        "data": barcode.data.decode(),
                        "position": {"x": x, "y": y_adjusted, "w": w, "h": h}
                    })
                best_thresh_image = thresh
                break

        # Save annotated ROI with timestamp to avoid overwrites
        cv2.imwrite(f'barcode_detection.jpg', img if results else gray)
        return results
    except Exception as e:
        logging.error(f"Barcode detection failed: {e}")
        return []


def extract_column_stitch(video_path):
    """Extract and stitch vertical columns from video frames."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video.")
        
        column_buffer = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            mid = frame.shape[1] // 2
            # -1:mid+1
            column = frame[:, mid-7:mid+6, :]
            column_buffer.append(column)
        cap.release()
        
        if not column_buffer:
            raise ValueError("No valid frames found in video.")
        
        return np.hstack(column_buffer)

    except Exception as e:
        logging.error(f"Column stitching failed: {e}")
        return np.array([])



def clean_ocr_barcode_pipeline(video_path, display=True):
    """Main pipeline to process video and extract barcode, batch, and expiry info."""
    try:
        start = time.time()
        stitched = extract_column_stitch(video_path)
        if stitched.size == 0:
            raise ValueError("Failed to extract stitched image from video.")
        
        enhanced = enhance_contrast(stitched)
        
        # Detect barcodes first; pass them into OCR step
        barcodes = detect_barcodes(enhanced)
        ocr_info = detect_batch_and_expiry(enhanced, barcodes=barcodes)
        line_info = detect_longest_line(enhanced, draw=True)


        # Annotate final result
        annotated = enhanced.copy()        
        # Add line info text if available
        if line_info.get("length_px") is not None:
            cv2.putText(
                annotated,
                f"Longest line: {line_info['length_px']:.1f}px @ {line_info['angle_deg']:.1f}°",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )
        
        colors = {
            "batch_value": (0, 255, 0),
            "expiry_value": (0, 255, 0)
        }

        for key, color in colors.items():
            if ocr_info.get(key):
                pos = ocr_info[key]["position"]
                x, y, w, h = pos["x"], pos["y"], pos["w"], pos["h"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = "Batch No" if "batch" in key else "Expiry"
                cv2.putText(annotated, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(annotated, f"Barcodes: {len(barcodes)}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if ocr_info.get("batch_value") == None or ocr_info.get("expiry_value") == None:
            cv2.putText(annotated, "Batch/Expiry not alligned", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else :
            cv2.putText(annotated, "Batch/Expiry aligned", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Save final annotated image
        
        end = time.time()
        # Add processing time and barcode count
        cv2.putText(annotated, f"Time: {end - start:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f'final_annotated.jpg', annotated)
        cv2.imwrite(f'panorama.jpg', stitched)
        # Display result if enabled
        if display:
            cv2.namedWindow("Final Annotated Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Final Annotated Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        return {
            "processing_time": round(end - start, 2),
            "barcodes": barcodes,
            "batch_expiry": ocr_info,
            "longest_line": line_info
        }
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return {
            "processing_time": 0.0,
            "barcodes": [],
            "batch_expiry": {
                "batch_label": None,
                "batch_value": None,
                "expiry_label": None,
                "expiry_value": None
            },
            "longest_line": {
                "length_px": None,
                "angle_deg": None,
                "endpoints": None,
                "roi_used": None
            }
        }

def select_rois_and_save(image, save_path="roi_config.json", barcodes=None):
    """Let user select ROI for batch and expiry and save to config.
       If a barcode is available, store ROIs relative to the barcode (normalized)."""
    rois = cv2.selectROIs("Select ROIs (Batch first, then Expiry)", image)
    cv2.destroyAllWindows()
    
    if len(rois) < 2:
        logging.warning("Less than two ROIs selected. Skipping ROI save.")
        return

    # Prepare absolute boxes from selections
    batch_abs = {"x": int(rois[0][0]), "y": int(rois[0][1]), "w": int(rois[0][2]), "h": int(rois[0][3])}
    expiry_abs= {"x": int(rois[1][0]), "y": int(rois[1][1]), "w": int(rois[1][2]), "h": int(rois[1][3])}

    config = {}
    # If we have barcodes, choose a reference and save relative + meta
    ref_barcode, ref_idx = choose_reference_barcode(barcodes or [])
    if ref_barcode is not None:
        ref = ref_barcode["position"]
        try:
            batch_rel  = to_relative_box(batch_abs, ref)
            expiry_rel = to_relative_box(expiry_abs, ref)
            config = {
                "meta": {
                    "anchor": "barcode",
                    "ref_barcode_index": ref_idx,
                    "relative": True,
                    "note": "ROIs normalized to chosen barcode box (dx,dy,w,h / (bw,bh))."
                },
                # Keep absolute copies for fallback/backward-compat
                "batch_roi": batch_abs,
                "expiry_roi": expiry_abs,
                # Relative (preferred)
                "batch_roi_rel": batch_rel,
                "expiry_roi_rel": expiry_rel
            }
        except Exception as e:
            logging.warning(f"Failed to compute relative ROIs; saving absolute only. Error: {e}")
            config = {
                "meta": {
                    "anchor": "none",
                    "relative": False
                },
                "batch_roi": batch_abs,
                "expiry_roi": expiry_abs
            }
    else:
        logging.warning("No barcode found while saving ROIs; saving absolute pixel ROIs.")
        config = {
            "meta": {
                "anchor": "none",
                "relative": False
            },
            "batch_roi": batch_abs,
            "expiry_roi": expiry_abs
        }

    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Saved ROIs to {save_path}")

def detect_batch_and_expiry(
    img,
    roi_config_path="roi_config.json",
    barcodes=None,
    block_sizes=range(7, 21, 2),
    constants=range(7, 21, 2)
):
    """Detect batch and expiry info in manually selected ROI(s).
       If ROI config contains *_roi_rel and a barcode is provided, resolve to absolute via barcode."""
    try:
        H, W = img.shape[:2]
        if not os.path.exists(roi_config_path):
            logging.warning("ROI config not found. Using full image halves.")
            roi_config = {
                "meta": {"anchor": "none", "relative": False},
                "batch_roi": {"x": 0, "y": 0, "w": W, "h": H // 2},
                "expiry_roi": {"x": 0, "y": H // 2, "w": W, "h": H // 2}
            }
        else:
            with open(roi_config_path) as f:
                roi_config = json.load(f)

        # Resolve reference barcode if available/desired
        ref_barcode = None
        if barcodes:
            # Respect stored index if present; otherwise pick largest
            idx = roi_config.get("meta", {}).get("ref_barcode_index", None)
            if isinstance(idx, int) and 0 <= idx < len(barcodes):
                ref_barcode = barcodes[idx]
            else:
                ref_barcode, _ = choose_reference_barcode(barcodes)

        # Resolve ROIs (prefer relative if available and ref barcode found)
        def resolve_roi(key_abs, key_rel):
            if key_rel in roi_config and ref_barcode is not None:
                try:
                    rel = roi_config[key_rel]
                    ref = ref_barcode["position"]
                    return from_relative_box(rel, ref, W, H)
                except Exception as e:
                    logging.warning(f"Failed to resolve {key_rel} via barcode; falling back to absolute. Error: {e}")
            # Fallback to absolute if present
            if key_abs in roi_config:
                return roi_config[key_abs]
            # Last resort: full image
            return {"x": 0, "y": 0, "w": W, "h": H}

        batch_roi  = resolve_roi("batch_roi",  "batch_roi_rel")
        expiry_roi = resolve_roi("expiry_roi", "expiry_roi_rel")

        # ---- OCR over the resolved ROIs ----
        results = {
            "batch_value": None,
            "expiry_value": None
        }

        for key, roi in [("batch_roi", batch_roi), ("expiry_roi", expiry_roi)]:
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            roi_img = img[y:y+h, x:x+w]
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            found = False
            for block_size in block_sizes:
                if found: break
                for C in constants:
                    if found: break
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, block_size, C
                    )
                    data = pytesseract.image_to_data(
                        thresh, config="--psm 11", output_type=pytesseract.Output.DICT
                    )
                    n_boxes = len(data["text"])
                    # print(f"Detected words in {key}: {data['text']}")
                    for i in range(n_boxes):
                        word = data["text"][i].strip().lower()
                        word2 = data["text"][i+1].strip().lower() if i+1 < n_boxes else ""
                        if not word:
                            continue

                        rx, ry, rw, rh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                        abs_x, abs_y = rx + x, ry + y

                        if key == "batch_roi" and results["batch_value"] is None:
                            # 24760|24/60
                            if re.search(r"(?i)\b(24760|24/60\.?)\b", word):
                                results["batch_value"] = {
                                    "value": word,
                                    "position": {"x": abs_x, "y": abs_y, "w": rw, "h": rh}
                                }
                                found = True
                                break

                        elif key == "expiry_roi" and results["expiry_value"] is None:
                            combined = f"{word}{word2}"
                            # 11/2026|11-2026
                            if re.search(r'(?i)\b(11/2026|11-2026|11/202/\.?)\b', combined):
                                results["expiry_value"] = {
                                    "value": combined,
                                    "position": {"x": abs_x, "y": abs_y, "w": rw, "h": rh}
                                }
                                found = True
                                break

        return results

    except Exception as e:
        logging.error(f"Batch/Expiry ROI detection failed: {e}")
        return {
            "batch_value": None,
            "expiry_value": None
        }


# # temp-07282025192121-0000
# # temp-07282025182130-0000
# # temp-07282025194032-0000
# # temp-07292025194937-0000
# temp-07312025122712-0000
# temp-07312025123034-0000
# temp-07312025123046-0000
# temp-07312025123452-0000
import pathlib
import datetime
def record_flir_camera_to_avi(
    num_frames: int = 150,
    fps: float = 30.0,
    out_dir: str = "test_videos",
    basename: str = "flir",
    timeout_ms: int = 1000
) -> str:
    """
    Auto-detect the first FLIR camera via PySpin, capture `num_frames`, and save to an .avi.
    Returns the output file path.

    - Handles Mono and Bayer formats (converts to BGR for OpenCV).
    - Creates `out_dir` if needed.
    - Cleans up all PySpin resources, even on errors.
    """
    import os, pathlib, cv2, time
    from datetime import datetime
    import PySpin

    system = None
    cam_list = None
    cam = None
    img = None
    writer = None
    processor = None
    out_path = ""

    try:
        # ── discover camera ───────────────────────────────────────────────
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() == 0:
            raise RuntimeError("No FLIR camera found.")
        cam = cam_list.GetByIndex(0)

        # ── init + acquisition mode ───────────────────────────────────────
        cam.Init()
        if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
            cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        else:
            raise RuntimeError("AcquisitionMode isn't writable right now.")

        # Optional: set frame rate if available (many cameras require turning off auto)
        try:
            if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRateEnable.SetValue(True)
            if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                cam.AcquisitionFrameRate.SetValue(float(fps))
        except Exception:
            # Not critical; continue with camera's internal FPS
            pass

        # ── start capture ─────────────────────────────────────────────────
        cam.BeginAcquisition()
        processor = PySpin.ImageProcessor()
        # processor.SetColorProcessing(PySpin.HQ_LINEAR)

        # ── prepare writer after first valid frame (we need width/height) ─
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%m%d%Y%H%M%S")
        out_path = os.path.join(out_dir, f"{basename}-{ts}-{num_frames:04d}.avi")

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # try "XVID" if needed
        frames_written = 0

        while frames_written < num_frames:
            img = cam.GetNextImage(timeout_ms)
            if img.IsIncomplete():
                img.Release()
                img = None
                continue

            # Convert image to a format OpenCV can write
            pf = img.GetPixelFormat()
            try:
                # if pf in (PySpin.PixelFormat_Mono8, PySpin.PixelFormat_Mono16):
                #     # Mono → uint8 → BGR for a consistent AVI
                #     nv = img.GetNDArray()
                #     if nv.dtype != 'uint8':
                #         nv = (nv / 256).astype('uint8')  # scale 16→8 if needed
                #     frame = cv2.cvtColor(nv, cv2.COLOR_GRAY2BGR)
                # else:
                #     # Color/Bayer → convert to BGR8
                #     conv = processor.Convert(img, PySpin.PixelFormat_BGR8)
                #     frame = conv.GetNDArray()
                conv = processor.Convert(img, PySpin.PixelFormat_BGR8)
                frame = conv.GetNDArray()
            finally:
                img.Release()
                img = None

            # Init writer once we know the exact size
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=True)
                if not writer.isOpened():
                    raise RuntimeError("Failed to open VideoWriter (try codec='XVID').")

            writer.write(frame)
            frames_written += 1

        if writer is not None:
            writer.release(); writer = None

        if frames_written == 0:
            raise RuntimeError("No frames captured from camera.")

        return out_path

    finally:
        # ── robust cleanup (matches your style) ───────────────────────────
        try:
            if writer is not None:
                writer.release()
        except:
            pass
        try:
            if img is not None:
                img.Release()
        except:
            pass
        try:
            if cam is not None:
                cam.EndAcquisition()
        except:
            pass
        try:
            if cam is not None:
                cam.DeInit()
        except:
            pass
        try:
            if cam_list is not None:
                cam_list.Clear()
        except:
            pass
        try:
            if cam is not None:
                del cam
            if cam_list is not None:
                del cam_list
        except:
            pass
        try:
            if system is not None:
                system.ReleaseInstance()
        except:
            pass

# if __name__ == "__main__":
#     try:
#         while (True):
#             mode = input("Select mode: [1] Run Pipeline, [2] Select OCR ROIs, [3] Select Line ROI: ").strip()
#             video_file = "test_videos\\temp-07312025123452-0000.avi"
#             if mode == "0":
#                 break
#             elif mode == "2":
#                 test_img = extract_column_stitch(video_file)
#                 if test_img.size > 0:
#                     enhanced = enhance_contrast(test_img)
#                     bcs = detect_barcodes(enhanced)
#                     if not bcs:
#                         logging.warning("No barcode detected. ROIs will be saved in absolute pixels.")
#                     select_rois_and_save(enhanced, barcodes=bcs)  # saves relative if barcode available
#             elif mode == "3":
#                 test_img = extract_column_stitch(video_file)
#                 if test_img.size > 0:
#                     select_single_roi_and_save(test_img, key="line_roi")  # saves line_roi
#             else:
#                 data = clean_ocr_barcode_pipeline(video_file)
#                 print(json.dumps(data, indent=2))
#     except Exception as e:
#         logging.error(f"Main execution failed: {e}")

# # temp-07282025192121-0000
# # temp-07282025182130-0000
# # temp-07282025194032-0000
# # temp-07292025194937-0000
# temp-07312025122712-0000
# temp-07312025123034-0000
# temp-07312025123046-0000
# temp-07312025123452-0000
# temp-08012025122714-0000
if __name__ == "__main__":
    try:
        while True:
            mode = input("Select mode: [1] Run Pipeline, [2] Select OCR ROIs, [3] Select Line ROI, [4] Capture N Frames + Run Pipeline (cam) [0] Exit: ").strip()

            # default video file (existing behavior)
            video_file = "test_videos\\temp-08072025154458-0000.avi"

            if mode == "0":
                break

            elif mode == "2":
                test_img = extract_column_stitch(video_file)
                if test_img.size > 0:
                    enhanced = enhance_contrast(test_img)
                    bcs = detect_barcodes(enhanced)
                    if not bcs:
                        logging.warning("No barcode detected. ROIs will be saved in absolute pixels.")
                    select_rois_and_save(enhanced, barcodes=bcs)  # saves relative if barcode available

            elif mode == "3":
                test_img = extract_column_stitch(video_file)
                if test_img.size > 0:
                    select_single_roi_and_save(test_img, key="line_roi")  # saves line_roi

            elif mode == "4":
                # --- new: capture from camera, then run pipeline on the captured clip ---
                try:
                    num_frames = 200
                    fps = 84.16

                    recorded_file = record_flir_camera_to_avi(
                        num_frames=num_frames,
                        fps=fps,
                    )
                    print("Saved:", recorded_file)
                    # Reuse Mode 1 behavior:
                    data = clean_ocr_barcode_pipeline(recorded_file)
                    print(json.dumps(data, indent=2))

                except Exception as rec_err:
                    logging.error(f"Camera capture failed: {rec_err}")

            else:
                # Mode "1" and any other input defaults to "Run Pipeline" on the given file
                data = clean_ocr_barcode_pipeline(video_file)
                print(json.dumps(data, indent=2))
    except Exception as e:
        logging.error(f"Main execution failed: {e}")