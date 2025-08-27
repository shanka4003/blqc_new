# # Let's load the provided BMP image, detect the laser line ridge column-by-column,
# # fit a baseline, compute deflection, flag outliers, and export visuals + a CSV.

# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# # 1) Load image
# # 22157381_20250810132419.bmp
# # 22157381_20250810132159
# # 22157381_20250810164819
# # 22157381_20250810164852
# path = "line_scan_data/22157381_20250810164819.bmp"
# im = Image.open(path).convert("RGB")
# arr = np.asarray(im).astype(np.float32)
# R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
# red = R - np.maximum(G, B)
# red -= red.min()
# if red.max() > 0:
#     red /= red.max()

# h, w = red.shape

# # For each row, pick the brightest column (within an 85th percentile mask)
# ridge_x = np.zeros(h, dtype=np.float32)
# confidence = np.zeros(h, dtype=np.float32)
# for y in range(h):
#     row = red[y, :]
#     thr = np.percentile(row, 85)
#     masked = np.where(row >= thr, row, 0.0)
#     x = np.argmax(masked)
#     ridge_x[y] = float(x)
#     confidence[y] = float(masked[x])

# # Smooth the horizontal position along rows
# win = max(5, h // 200)
# if win % 2 == 0: win += 1
# kernel = np.ones(win, dtype=np.float32) / win
# ridge_x_smooth = np.convolve(ridge_x, kernel, mode="same")

# # Fit a 1st-order (vertical line ~ constant x) baseline x(y) = a*y + b
# y_coords = np.arange(h, dtype=np.float32)
# deg = 1
# coef = np.polyfit(y_coords, ridge_x_smooth, deg)
# baseline = np.polyval(coef, y_coords)

# # Deflection in pixels (horizontal)
# deflection = ridge_x - baseline

# # Robust outlier score (MAD)
# median = np.median(deflection)
# mad = np.median(np.abs(deflection - median)) + 1e-6
# modified_z = 0.6745 * (deflection - median) / mad

# # Bubble candidates: horizontal kinks where |z| > 3 and confidence not tiny
# candidates = (np.abs(modified_z) > 3) & (confidence > np.percentile(confidence, 75))
# print("Bubble candidates found:", np.sum(candidates))
# print("Deflection stats: mean =", np.mean(deflection), "std =", np.std(deflection))
# # Save CSV
# df = pd.DataFrame({
#     "y": y_coords.astype(int),
#     "x_detected": ridge_x,
#     "baseline_x": baseline,
#     "deflection_x_px": deflection,
#     "modified_z": modified_z,
#     "confidence": confidence
# })
# csv_path = "laser_deflection_vertical_profile.csv"
# df.to_csv(csv_path, index=False)

# # Overlay: draw detected vertical ridge and candidate points
# plt.figure()
# plt.imshow(red, cmap="gray")
# plt.plot(ridge_x, y_coords, linewidth=1)  # x vs y
# plt.scatter(ridge_x[candidates], y_coords[candidates], s=50)
# overlay_path = "annotated_vertical_overlay.png"
# plt.title("Vertical Red Ridge (dots = suspected bubble deflections)")
# plt.gca().invert_yaxis()  # keep image-like coordinates
# plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
# plt.close()

# # Deflection plot: horizontal deflection vs height
# plt.figure()
# plt.plot(deflection, y_coords, linewidth=1)
# plt.axvline(0, linestyle="--", linewidth=1)
# plt.title("Horizontal Deflection (px) along height")
# plt.xlabel("deflection_x (px)")
# plt.ylabel("y (row)")
# plt.gca().invert_yaxis()
# deflect_plot_path = "vertical_deflection_plot.png"
# plt.savefig(deflect_plot_path, dpi=150, bbox_inches="tight")
# plt.close()

# csv_path, overlay_path, deflect_plot_path

# Detect a horizontal laser line across columns, fit y(x), compute vertical deflection,
# flag outliers, and export visuals + a CSV.

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Load image
# 22157381_20250810132419.bmp
# 22157381_20250810132159
# 22157381_20250810164819
# 22157381_20250810164852

# 22157381_20250811185005.bmp
# 22157381_20250811184943
path = "line_scan_data/22157381_20250812161929.bmp"
im = Image.open(path).convert("RGB")
arr = np.asarray(im).astype(np.float32)
R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

# Emphasize the red channel vs. others and normalize to [0,1]
red = R - np.maximum(G, B)
red -= red.min()
if red.max() > 0:
    red /= red.max()

h, w = red.shape

# For each column, pick the brightest row (within an 85th percentile mask)
ridge_y = np.zeros(w, dtype=np.float32)
confidence = np.zeros(w, dtype=np.float32)
for x in range(w):
    col = red[:, x]
    thr = np.percentile(col, 95)
    masked = np.where(col >= thr, col, 0.0)
    y = int(np.argmax(masked))
    ridge_y[x] = float(y)
    confidence[x] = float(masked[y])

# Smooth the vertical position along columns
win = max(5, w // 200)
if win % 2 == 0:
    win += 1
kernel = np.ones(win, dtype=np.float32) / win
ridge_y_smooth = np.convolve(ridge_y, kernel, mode="same")

# Fit a 1st-order baseline: y(x) = a*x + b (horizontal line ~ constant y)
x_coords = np.arange(w, dtype=np.float32)
deg = 1
coef = np.polyfit(x_coords, ridge_y_smooth, deg)
baseline = np.polyval(coef, x_coords)

# Deflection in pixels (vertical)
deflection = ridge_y - baseline

# Robust outlier score (MAD)
median = np.median(deflection)
mad = np.median(np.abs(deflection - median)) + 1e-6
modified_z = 0.6745 * (deflection - median) / mad

# Bubble candidates: vertical kinks where |z| > 3 and confidence not tiny
candidates = (np.abs(modified_z) > 2) & (confidence > np.percentile(confidence, 65))
print("Bubble candidates found:", int(np.sum(candidates)))
print("Deflection stats: mean =", float(np.mean(deflection)), "std =", float(np.std(deflection)))

# Save CSV
df = pd.DataFrame({
    "x": x_coords.astype(int),
    "y_detected": ridge_y,
    "baseline_y": baseline,
    "deflection_y_px": deflection,
    "modified_z": modified_z,
    "confidence": confidence
})
csv_path = "laser_deflection_horizontal_profile.csv"
df.to_csv(csv_path, index=False)

# Overlay: draw detected horizontal ridge and candidate points
plt.figure()
plt.imshow(red, cmap="gray")
plt.plot(x_coords, ridge_y, linewidth=1)  # y vs x
plt.scatter(x_coords[candidates], ridge_y[candidates], s=50)
overlay_path = "annotated_horizontal_overlay.png"
plt.title("Horizontal Red Ridge (dots = suspected bubble deflections)")
plt.gca().invert_yaxis()  # image-like coordinates
plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
plt.close()

# Deflection plot: vertical deflection vs width (x)
plt.figure()
plt.plot(x_coords, deflection, linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Vertical Deflection (px) along width")
plt.xlabel("x (column)")
plt.ylabel("deflection_y (px)")
deflect_plot_path = "horizontal_deflection_plot.png"
plt.savefig(deflect_plot_path, dpi=150, bbox_inches="tight")
plt.close()

csv_path, overlay_path, deflect_plot_path
