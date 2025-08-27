import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Load images
img1 = cv2.imread("panorama.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("panorama_1.jpg", cv2.IMREAD_COLOR)

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ECC alignment
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)

try:
    (cv, warp_matrix) = cv2.findTransformECC(gray1, gray2, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
    aligned_img2 = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
except cv2.error as e:
    print("ECC alignment failed, using unaligned image")
    aligned_img2 = img2

# SSIM
# gray_aligned = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY)
# score, diff = ssim(gray1, gray_aligned, full=True)
# diff = (diff * 255).astype("uint8")

# # Thresholding to find differences
# thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY_INV)[1]

# Save results
cv2.imwrite("aligned_panorama.jpg", aligned_img2)
