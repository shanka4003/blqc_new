import cv2
import numpy as np
from PIL import Image

# Paths
p1 = "latest/panorama.jpg"
p2 = "latest/panorama_1.jpg"

# Load RGB and compute grayscale (OpenCV uses BGR by default)
im1 = cv2.imread(p1, cv2.IMREAD_COLOR)
im2 = cv2.imread(p2, cv2.IMREAD_COLOR)

# Re-detect barcode centers to recompute shift (for reproducibility in this cell)
def detect_barcode_center(img):
    # Try cv2 barcode API; fall back to gradient-based detection
    center = None
    box = None
    try:
        bd = cv2.barcode_BarcodeDetector()
        ok, decoded_info, decoded_type, corners = bd.detectAndDecode(img)
        if ok and corners is not None and len(corners) > 0:
            c = np.squeeze(corners[0])
            if c.shape == (4,2):
                box = c.astype(np.float32)
    except Exception:
        pass
    
    if box is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        ax = np.abs(sobelx)
        ax = cv2.GaussianBlur(ax, (9,9), 0)
        ax_norm = cv2.normalize(ax, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thr = cv2.threshold(ax_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 11))
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best = None
            best_score = -1
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (w, h), angle = rect
                if w < 20 or h < 20:
                    continue
                area = w * h
                ar = max(w, h) / (min(w, h) + 1e-6)
                score = area * ar
                if score > best_score:
                    best_score = score
                    best = cv2.boxPoints(rect).astype(np.float32)
            box = best
    
    if box is not None:
        center = box.mean(axis=0)
    return center

c1 = detect_barcode_center(im1)
c2 = detect_barcode_center(im2)

if c1 is None or c2 is None:
    raise RuntimeError("Could not detect barcodes in both images.")

# Compute translation to align image2 to image1
shift = c1 - c2  # dx, dy to move im2
M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])

# Apply translation
aligned2 = cv2.warpAffine(im2, M, (im1.shape[1], im1.shape[0]), flags=cv2.INTER_LINEAR)

# Compute intersection mask to avoid border artifacts
mask = np.zeros((im1.shape[0], im1.shape[1]), dtype=np.uint8)
mask[:] = 255
mask_shifted = cv2.warpAffine(mask, M, (im1.shape[1], im1.shape[0]), flags=cv2.INTER_NEAREST)
valid = cv2.bitwise_and(mask, mask_shifted)

# Difference map on the valid region only
diff = cv2.absdiff(im1, aligned2)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
diff_gray_masked = cv2.bitwise_and(diff_gray, diff_gray, mask=valid)
_, diff_thresh = cv2.threshold(diff_gray_masked, 30, 255, cv2.THRESH_BINARY)

# Save outputs
aligned_path = "aligned_by_barcode.jpg"
diff_path = "diff_after_barcode_align.jpg"
diff_thresh_path = "diff_thresholded_after_barcode_align.jpg"

cv2.imwrite(aligned_path, aligned2)
cv2.imwrite(diff_path, diff_gray_masked)
cv2.imwrite(diff_thresh_path, diff_thresh)

aligned_path, diff_path, diff_thresh_path
