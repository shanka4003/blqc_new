import sys, os, glob, json, tempfile
import cv2, numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QSpinBox, QCheckBox, QMessageBox,
    QProgressBar, QRadioButton, QButtonGroup, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
# ─── your full pipeline below ─────────────────────────────────────────────────

def compensate_with_flow(img1, img2, pyr_scale=0.5, levels=2, winsize=80):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g2, g1, None, pyr_scale, levels,
        winsize, 3, 5, 1.2, 0
    )
    half = flow * 0.5
    h, w = g1.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    map1_x = (gx + half[...,0]).astype(np.float32)
    map1_y = (gy + half[...,1]).astype(np.float32)
    warped1 = cv2.remap(img1, map1_x, map1_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    map2_x = (gx - half[...,0]).astype(np.float32)
    map2_y = (gy - half[...,1]).astype(np.float32)
    warped2 = cv2.remap(img2, map2_x, map2_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    return warped1, warped2

def blend_overlap(w1, w2, shift):
    h, w = w1.shape[:2]
    x0 = max(0, shift)
    x1 = min(w, shift + w)
    if x1 <= x0:
        return np.hstack((w1, w2)) if shift>=0 else np.hstack((w2, w1))

    ov_w = x1 - x0
    alpha = np.linspace(0, 1, ov_w).reshape(1, ov_w, 1)
    left = w1[:, :x0]
    m1   = w1[:, x0:x1].astype(np.float32)
    m2   = w2[:, (x0-shift):(x1-shift)].astype(np.float32)
    right= w2[:, (x1-shift):]
    mid = (m1*(1-alpha) + m2*alpha).astype(np.uint8)
    return np.concatenate([left, mid, right], axis=1)

def estimate_horizontal_shift_with_bf(img_ref, img_new, mean_shift, ratio_thresh=0.7):
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    kp_new, des_new = sift.detectAndCompute(img_new, None)
    if des_ref is None or des_new is None or not kp_ref or not kp_new:
        return -int(round(mean_shift))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_new, k=2)
    good = []
    for pair in matches:
        if len(pair)==2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good.append(pair[0])
    if not good:
        return -int(round(mean_shift))
    dx = [kp_new[m.trainIdx].pt[0] - kp_ref[m.queryIdx].pt[0] for m in good]
    shift = int(np.median(dx))
    m = int(round(mean_shift))
    if not (m*0.2 <= -shift <= m*1.8):
        shift = -m
    return shift

def refine_shift_with_flow(img_ref, img_new, initial_shift):
    h_ref, w_ref = img_ref.shape[:2]
    h_new, w_new = img_new.shape[:2]
    x_min = min(0, initial_shift)
    x_base = -x_min
    x_new_start = x_base + initial_shift
    start = int(round(max(x_base, x_new_start)))
    end   = int(round(min(x_base + w_ref, x_new_start + w_new)))
    if end <= start:
        return initial_shift
    ref_gray = cv2.cvtColor(img_ref[:, start - x_base:end - x_base], cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(img_new[:, (start-x_new_start):(start-x_new_start + (end-start))], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(new_gray, ref_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[...,0].flatten()
    if dx.size==0:
        return initial_shift
    delta = int(np.median(dx))
    return initial_shift + delta

def stitch_pair_by_shift(img_base, img_new, shift):
    h_b, w_b = img_base.shape[:2]
    h_n, w_n = img_new.shape[:2]
    x_min = min(0, shift)
    x_max = max(w_b, shift + w_n)
    pano_h = max(h_b, h_n)
    pano_w = x_max - x_min
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    x_b = -x_min
    pano[:h_b, x_b:x_b+w_b] = img_base
    x_n = x_b + shift
    pano[:h_n, x_n:x_n+w_n] = img_new
    start = max(x_b, x_n); end = min(x_b+w_b, x_n+w_n)
    if end <= start:
        return pano
    base_crop = img_base[:, (start-x_b):(end-x_b)]
    new_crop  = img_new[:,  (start-x_n):(end-x_n)]
    warped_base, warped_new = compensate_with_flow(base_crop, new_crop)
    local_shift = x_n - start
    blended = blend_overlap(warped_base, warped_new, local_shift)
    pano[:, start:end] = blended
    return pano

def stitch_strips_with_flow(strips):
    pano = strips[0]
    prev = strips[0]
    prevshifts = []
    os.makedirs("results", exist_ok=True)
    for i in range(1, len(strips)):
        cur = strips[i]
        mean_s = np.mean(prevshifts[2:-1]) if len(prevshifts)>3 else 90
        bf  = estimate_horizontal_shift_with_bf(prev, cur, mean_s)
        shift = refine_shift_with_flow(prev, cur, -bf)
        prevshifts.append(shift)
        pano = stitch_pair_by_shift(pano, cur, sum(prevshifts))
        prev = cur
        cv2.imwrite(f"results/pano_step_{i}.jpg", pano)
    return pano

def select_points(img_path, pts_path, grid_rows=20, grid_cols=20,
                  max_display=(1280, 720)):
    orig = cv2.imread(img_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot open {img_path}")
    H, W = orig.shape[:2]
    grid_img = orig.copy()
    color, thickness = (200,200,200), 1
    for i in range(1, grid_cols):
        x = int(W * i / grid_cols)
        cv2.line(grid_img, (x, 0), (x, H), color, thickness)
    for j in range(1, grid_rows):
        y = int(H * j / grid_rows)
        cv2.line(grid_img, (0, y), (W, y), color, thickness)
    maxW, maxH = max_display
    scale = min(maxW / W, maxH / H, 1.0)
    disp = cv2.resize(grid_img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    points = []
    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
            orig_x = int(x/scale); orig_y = int(y/scale)
            points.append((orig_x, orig_y))
            cv2.circle(disp, (x, y), 5, (0,0,255), -1)
            cv2.imshow(win, disp)

    win = "Select 6 points (3 top, 3 bottom)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) != 6:
        raise ValueError(f"You clicked {len(points)} points—I need exactly 6.")
    with open(pts_path, "w") as f:
        json.dump(points, f, indent=2)

def load_points(pts_path):
    with open(pts_path, "r") as f:
        pts = json.load(f)
    return np.array(pts[:3], float), np.array(pts[3:], float)

def flatten_strip(img, top_pts, bot_pts, out_h):
    top_fit = np.polyfit(top_pts[:,0], top_pts[:,1], 2)
    bot_fit = np.polyfit(bot_pts[:,0], bot_pts[:,1], 2)
    def E(f,x): return f[0]*x**2 + f[1]*x + f[2]
    x0, x2 = top_pts[0,0], top_pts[2,0]
    width = int(abs(x2 - x0))
    x_vals = np.linspace(x0, x2, width).astype(int)
    strip = np.zeros((out_h, width, 3), np.uint8)
    for i, x in enumerate(x_vals):
        y_t = E(top_fit, x); y_b = E(bot_fit, x)
        ys = np.linspace(y_t, y_b, out_h).astype(int)
        for j, y in enumerate(ys):
            if 0<=x<img.shape[1] and 0<=y<img.shape[0]:
                strip[j,i] = img[y,x]
    return strip
# def flatten_strip(img, top_pts, bot_pts, out_h):
#     # 1) fit quadratics
#     top_fit = np.polyfit(top_pts[:,0], top_pts[:,1], 2)
#     bot_fit = np.polyfit(bot_pts[:,0], bot_pts[:,1], 2)

#     # 2) determine x-range
#     x0, x2 = int(round(top_pts[0,0])), int(round(top_pts[2,0]))
#     width = x2 - x0
#     x_vals = np.linspace(x0, x2, width, dtype=np.float32)

#     # 3) compute float‐y for top & bottom
#     top_y = np.polyval(top_fit, x_vals)
#     bot_y = np.polyval(bot_fit, x_vals)

#     # 4) build map_x/map_y
#     # map_x is the same x for every row
#     map_x = np.tile(x_vals, (out_h,1))
#     # map_y interpolates between top_y and bot_y along each row
#     v = np.linspace(0,1,out_h, dtype=np.float32)[:,None]
#     map_y = (1-v)*top_y + v*bot_y
#     map_x = map_x.astype(np.float32)
#     map_y = map_y.astype(np.float32)

#     # 5) remap with bilinear interpolation
#     strip = cv2.remap(img, map_x, map_y,
#                       interpolation=cv2.INTER_LINEAR,
#                       borderMode=cv2.BORDER_REFLECT)
#     return strip

# ─── end pipeline ─────────────────────────────────────────────────────────────

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, input_type, in_path, pts_path, height, no_stitch):
        super().__init__()
        self.input_type = input_type
        self.in_path = in_path
        self.pts_path = pts_path
        self.height = height
        self.no_stitch = no_stitch

    def run(self):
        try:
            top, bot = load_points(self.pts_path)
            strips = []

            if self.input_type == "Frames Folder":
                files = sorted(glob.glob(os.path.join(self.in_path, "*.*")))
                imgs = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
                total = len(imgs)
                if not total:
                    raise RuntimeError("No images in folder.")
                for i, fn in enumerate(imgs):
                    img = cv2.imread(fn)
                    # img = cv2.flip(img, 1)  # flip Horizontally if needed
                    if img is None: continue
                    strips.append(flatten_strip(img, top, bot, self.height))
                    self.progress.emit(int(100*(i+1)/total))

            else:  # Video
                cap = cv2.VideoCapture(self.in_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                for i in range(total):
                    ret, frame = cap.read()
                    if not ret: break
                    # frame = cv2.flip(frame, 1)  # flip Horizontally if needed
                    strips.append(flatten_strip(frame, top, bot, self.height))
                    self.progress.emit(int(100*(i+1)/total))
                cap.release()
                if not strips:
                    raise RuntimeError("Couldn't read any frames.")

            # stitch or concat
            if self.no_stitch:
                pano = cv2.hconcat(strips)
            else:
                pano = stitch_strips_with_flow(strips)

            self.finished.emit(pano)

        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stretch Dashboard")
        self.resize(900, 700)

        # Input type
        r1 = QRadioButton("Frames Folder"); r2 = QRadioButton("Video File")
        r1.setChecked(True)
        self.rg = QButtonGroup(self)
        self.rg.addButton(r1); self.rg.addButton(r2)
        self.rg.buttonToggled.connect(self.on_input_type_changed)

        # Input path
        self.inputEdit = QLineEdit()
        self.btnBrowse = QPushButton("Browse…")
        self.btnBrowse.clicked.connect(self.browse)

        # Points JSON path
        self.ptsEdit = QLineEdit()
        self.ptsEdit.setReadOnly(True)

        # Frame selector
        self.frameFileEdit = QLineEdit()
        self.frameFileEdit.setPlaceholderText("Choose image in folder…")
        self.btnFrameBrowse = QPushButton("Browse Frame")
        self.btnFrameBrowse.clicked.connect(self.browse_frame)

        self.frameNumSpin = QSpinBox()
        self.frameNumSpin.setRange(0, 1000000)
        self.frameNumSpin.setValue(0)

        # Other controls
        self.heightSpin = QSpinBox()
        self.heightSpin.setRange(50, 2000)
        self.heightSpin.setValue(400)
        self.chkNoStitch = QCheckBox("No‐Stitch (concat only)")
        btnPick = QPushButton("Select 6 Points")
        btnPick.clicked.connect(self.pick_points)
        btnRun  = QPushButton("Run Stretch")
        btnRun.clicked.connect(self.run_process)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.imageLabel = QLabel("Result will appear here")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Layout
        topLay = QHBoxLayout()
        topLay.addWidget(r1); topLay.addWidget(r2)

        pathLay = QHBoxLayout()
        pathLay.addWidget(self.inputEdit); pathLay.addWidget(self.btnBrowse)

        ptsLay = QHBoxLayout()
        ptsLay.addWidget(QLabel("Points JSON:")); ptsLay.addWidget(self.ptsEdit)

        frameLay = QHBoxLayout()
        frameLay.addWidget(QLabel("Frame:"))
        frameLay.addWidget(self.frameFileEdit)
        frameLay.addWidget(self.btnFrameBrowse)
        frameLay.addWidget(QLabel("Frame #"))
        frameLay.addWidget(self.frameNumSpin)

        ctrlLay = QHBoxLayout()
        ctrlLay.addWidget(QLabel("Height:")); ctrlLay.addWidget(self.heightSpin)
        ctrlLay.addWidget(self.chkNoStitch)
        ctrlLay.addWidget(btnPick); ctrlLay.addWidget(btnRun)

        mainLay = QVBoxLayout()
        mainLay.addLayout(topLay)
        mainLay.addLayout(pathLay)
        mainLay.addLayout(ptsLay)
        mainLay.addLayout(frameLay)
        mainLay.addLayout(ctrlLay)
        mainLay.addWidget(self.progress)
        mainLay.addWidget(self.imageLabel)

        container = QWidget()
        container.setLayout(mainLay)
        self.setCentralWidget(container)

        self.on_input_type_changed()  # init UI state

    @property
    def input_type(self):
        return "Frames Folder" if self.rg.buttons()[0].isChecked() else "Video File"

    def browse(self):
        if self.input_type == "Frames Folder":
            d = QFileDialog.getExistingDirectory(self, "Choose Frames Folder")
            if d: self.inputEdit.setText(d)
        else:
            f, _ = QFileDialog.getOpenFileName(
                self, "Choose Video File",
                filter="Videos (*.mp4 *.avi *.mov)"
            )
            if f: self.inputEdit.setText(f)
        self.update_pts_path()

    def on_input_type_changed(self, *_):
        if self.input_type == "Frames Folder":
            self.inputEdit.setPlaceholderText("Select folder of frames")
            self.frameFileEdit.setEnabled(True)
            self.btnFrameBrowse.setEnabled(True)
            self.frameNumSpin.setEnabled(False)
        else:
            self.inputEdit.setPlaceholderText("Select a video file")
            self.frameFileEdit.setEnabled(False)
            self.btnFrameBrowse.setEnabled(False)
            self.frameNumSpin.setEnabled(True)
        self.update_pts_path()

    def update_pts_path(self):
        path = self.inputEdit.text().strip()
        if not path:
            self.ptsEdit.clear()
            return
        if self.input_type == "Frames Folder":
            pts = os.path.join(path, "points.json")
        else:
            d, fn = os.path.split(path)
            base, _ = os.path.splitext(fn)
            pts = os.path.join(d, f"{base}_points.json")
        self.ptsEdit.setText(pts)

    def browse_frame(self):
        base = self.inputEdit.text().strip()
        if not os.path.isdir(base):
            QMessageBox.warning(self, "Error", "Select a valid frames folder first.")
            return
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select Frame Image", base,
            filter="Images (*.jpg *.jpeg *.png)"
        )
        if fn:
            self.frameFileEdit.setText(fn)

    def pick_points(self):
        path = self.inputEdit.text().strip()
        pts  = self.ptsEdit.text().strip()
        if not path or not pts:
            QMessageBox.warning(self, "Error", "Choose input & points path first.")
            return

        try:
            if self.input_type == "Frames Folder":
                # use chosen frame or first image
                fn = self.frameFileEdit.text().strip()
                if not fn:
                    imgs = sorted(glob.glob(os.path.join(path, "*.*")))
                    fn = next((i for i in imgs if i.lower().endswith((".jpg",".jpeg",".png"))), None)
                if not fn or not os.path.isfile(fn):
                    raise RuntimeError("No valid frame image found.")
                select_points(fn, pts)

            else:
                # video: grab specified frame number
                idx = self.frameNumSpin.value()
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError(f"Couldn’t grab frame {idx}.")
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                tmp.close()
                select_points(tmp.name, pts)
                os.unlink(tmp.name)

            QMessageBox.information(self, "Saved", f"points.json →\n{pts}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_process(self):
        path = self.inputEdit.text().strip()
        pts  = self.ptsEdit.text().strip()
        if not os.path.exists(path) or not os.path.isfile(pts):
            QMessageBox.warning(self, "Error", "Ensure input & points.json exist.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.thread = ProcessThread(
            self.input_type, path, pts,
            self.heightSpin.value(),
            self.chkNoStitch.isChecked()
        )
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_done)
        self.thread.error.connect(self.on_error)
        self.thread.start()


import sys
import os
import glob
import json
import tempfile
import cv2
import numpy as np
import PySpin
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QFileDialog, QHBoxLayout, QVBoxLayout, QSpinBox, QCheckBox, QMessageBox,
    QProgressBar, QRadioButton, QButtonGroup, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ─── your full pipeline below ─────────────────────────────────────────────────

def compensate_with_flow(img1, img2, pyr_scale=0.5, levels=2, winsize=80):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        g2, g1, None, pyr_scale, levels,
        winsize, 3, 5, 1.2, 0
    )
    half = flow * 0.5
    h, w = g1.shape
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))

    map1_x = (gx + half[...,0]).astype(np.float32)
    map1_y = (gy + half[...,1]).astype(np.float32)
    warped1 = cv2.remap(img1, map1_x, map1_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)

    map2_x = (gx - half[...,0]).astype(np.float32)
    map2_y = (gy - half[...,1]).astype(np.float32)
    warped2 = cv2.remap(img2, map2_x, map2_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
    return warped1, warped2

def blend_overlap(w1, w2, shift):
    h, w = w1.shape[:2]
    x0 = max(0, shift)
    x1 = min(w, shift + w)
    if x1 <= x0:
        return np.hstack((w1, w2)) if shift>=0 else np.hstack((w2, w1))

    ov_w = x1 - x0
    alpha = np.linspace(0, 1, ov_w).reshape(1, ov_w, 1)
    left = w1[:, :x0]
    m1   = w1[:, x0:x1].astype(np.float32)
    m2   = w2[:, (x0-shift):(x1-shift)].astype(np.float32)
    right= w2[:, (x1-shift):]
    mid = (m1*(1-alpha) + m2*alpha).astype(np.uint8)
    return np.concatenate([left, mid, right], axis=1)

def estimate_horizontal_shift_with_bf(img_ref, img_new, mean_shift, ratio_thresh=1.3):
    sift = cv2.SIFT_create()
    kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
    kp_new, des_new = sift.detectAndCompute(img_new, None)
    if des_ref is None or des_new is None or not kp_ref or not kp_new:
        return -int(round(mean_shift))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_ref, des_new, k=2)
    good = []
    for pair in matches:
        if len(pair)==2 and pair[0].distance < ratio_thresh * pair[1].distance:
            good.append(pair[0])
    if not good:
        return -int(round(mean_shift))
    dx = [kp_new[m.trainIdx].pt[0] - kp_ref[m.queryIdx].pt[0] for m in good]
    shift = int(np.median(dx))
    m = int(round(mean_shift))
    if not (m*0.6 <= -shift <= m*1.4):
        shift = -m
    return shift

def refine_shift_with_flow(img_ref, img_new, initial_shift):
    h_ref, w_ref = img_ref.shape[:2]
    h_new, w_new = img_new.shape[:2]
    x_min = min(0, initial_shift)
    x_base = -x_min
    x_new_start = x_base + initial_shift
    start = int(round(max(x_base, x_new_start)))
    end   = int(round(min(x_base + w_ref, x_new_start + w_new)))
    if end <= start:
        return initial_shift
    ref_gray = cv2.cvtColor(img_ref[:, start - x_base:end - x_base], cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(img_new[:, (start-x_new_start):(start-x_new_start + (end-start))], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(new_gray, ref_gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = flow[...,0].flatten()
    if dx.size==0:
        return initial_shift
    delta = int(np.median(dx))
    return initial_shift + delta

def stitch_pair_by_shift(img_base, img_new, shift):
    h_b, w_b = img_base.shape[:2]
    h_n, w_n = img_new.shape[:2]
    x_min = min(0, shift)
    x_max = max(w_b, shift + w_n)
    pano_h = max(h_b, h_n)
    pano_w = x_max - x_min
    pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
    x_b = -x_min
    pano[:h_b, x_b:x_b+w_b] = img_base
    x_n = x_b + shift
    pano[:h_n, x_n:x_n+w_n] = img_new
    start = max(x_b, x_n); end = min(x_b+w_b, x_n+w_n)
    if end <= start:
        return pano
    base_crop = img_base[:, (start-x_b):(end-x_b)]
    new_crop  = img_new[:,  (start-x_n):(end-x_n)]
    warped_base, warped_new = compensate_with_flow(base_crop, new_crop)
    local_shift = x_n - start
    blended = blend_overlap(warped_base, warped_new, local_shift)
    pano[:, start:end] = blended
    return pano

def stitch_strips_with_flow(strips):
    pano = strips[0]
    prev = strips[0]
    prevshifts = []
    os.makedirs("results", exist_ok=True)
    for i in range(1, len(strips)):
        cur = strips[i]
        mean_s = np.mean(prevshifts[2:-1]) if len(prevshifts)>3 else 20
        bf  = estimate_horizontal_shift_with_bf(prev, cur, mean_s)
        shift = refine_shift_with_flow(prev, cur, -bf)
        prevshifts.append(shift)
        pano = stitch_pair_by_shift(pano, cur, sum(prevshifts))
        prev = cur
        cv2.imwrite(f"results/pano_step_{i}.jpg", pano)
    return pano

def select_points(img_path, pts_path, grid_rows=20, grid_cols=20,
                  max_display=(1280, 720)):
    orig = cv2.imread(img_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot open {img_path}")
    H, W = orig.shape[:2]
    grid_img = orig.copy()
    color, thickness = (200,200,200), 1
    for i in range(1, grid_cols):
        x = int(W * i / grid_cols)
        cv2.line(grid_img, (x, 0), (x, H), color, thickness)
    for j in range(1, grid_rows):
        y = int(H * j / grid_rows)
        cv2.line(grid_img, (0, y), (W, y), color, thickness)
    maxW, maxH = max_display
    scale = min(maxW / W, maxH / H, 1.0)
    disp = cv2.resize(grid_img, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)

    points = []
    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
            orig_x = int(x/scale); orig_y = int(y/scale)
            points.append((orig_x, orig_y))
            cv2.circle(disp, (x, y), 5, (0,0,255), -1)
            cv2.imshow(win, disp)

    win = "Select 6 points (3 top, 3 bottom)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(points) != 6:
        raise ValueError(f"You clicked {len(points)} points—I need exactly 6.")
    with open(pts_path, "w") as f:
        json.dump(points, f, indent=2)

def make_strip_maps(top_pts, bot_pts, out_h):
    # Fit quadratics
    top_fit = np.polyfit(top_pts[:,0], top_pts[:,1], 2)
    bot_fit = np.polyfit(bot_pts[:,0], bot_pts[:,1], 2)

    # X-span of strip
    x0, x1 = int(round(top_pts[0,0])), int(round(top_pts[2,0]))
    width = abs(x1 - x0)
    x_vals = np.linspace(x0, x1, width, dtype=np.float32)

    # Y positions along top & bottom
    top_y = np.polyval(top_fit, x_vals)
    bot_y = np.polyval(bot_fit, x_vals)

    # build the two maps
    map_x = np.tile(x_vals, (out_h, 1))
    v = np.linspace(0, 1, out_h, dtype=np.float32)[:,None]
    map_y = (1 - v) * top_y + v * bot_y

    return map_x, map_y

def load_points(pts_path):
    with open(pts_path, "r") as f:
        pts = json.load(f)
    return np.array(pts[:3], float), np.array(pts[3:], float)

def flatten_strip(img, top_pts, bot_pts, out_h):
    top_fit = np.polyfit(top_pts[:,0], top_pts[:,1], 2)
    bot_fit = np.polyfit(bot_pts[:,0], bot_pts[:,1], 2)
    def E(f,x): return f[0]*x**2 + f[1]*x + f[2]
    x0, x2 = top_pts[0,0], top_pts[2,0]
    width = int(abs(x2 - x0))
    x_vals = np.linspace(x0, x2, width).astype(int)
    strip = np.zeros((out_h, width, 3), np.uint8)
    for i, x in enumerate(x_vals):
        y_t = E(top_fit, x); y_b = E(bot_fit, x)
        ys = np.linspace(y_t, y_b, out_h).astype(int)
        for j, y in enumerate(ys):
            if 0<=x<img.shape[1] and 0<=y<img.shape[0]:
                strip[j,i] = img[y,x]
    return strip

class ProcessThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self,
                 input_type, in_path, pts_path,
                 height, no_stitch,
                 roi=None, threshold=None, burst_size=None):
        super().__init__()
        self.input_type   = input_type
        self.in_path      = in_path
        self.pts_path     = pts_path
        self.height       = height
        self.no_stitch    = no_stitch
        self.roi          = roi
        self.threshold    = threshold
        self.burst_size   = burst_size

    def run(self):
        try:
            top, bot = load_points(self.pts_path)
            
            # ── AUTO MODE: loop forever, emitting on every change ─────────────────
            if self.input_type == "Auto Mode":
                system = PySpin.System.GetInstance()
                cams = system.GetCameras()
                if not cams:
                    raise RuntimeError("No FLIR camera found.")
                cam = cams[0]
                cam.Init()
                if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
                    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                else:
                    raise RuntimeError("AcquisitionMode isn't writable right now")                

                cam.BeginAcquisition()
                x, y, w, h = self.roi
                
                # nodemap = cam.GetNodeMap()
                try:
                    # endless loop: wait for change → burst → stitch → emit → repeat
                    while True:
                        prev_roi = None

                        # wait for ROI change
                        while True:
                            img = cam.GetNextImage()
                            if img.IsIncomplete():
                                img.Release()
                                continue

                            arr = img.GetNDArray()
                            patch = arr[y:y+h, x:x+w]
                            img.Release()

                            if prev_roi is not None:
                                diff = np.abs(patch.astype(int) - prev_roi.astype(int))
                                if diff.max() > self.threshold:
                                    break
                            prev_roi = patch.copy()

                        # burst-capture N frames
                        burst = []
                        self.progress.emit(0)
                        out_dir = os.path.join(os.getcwd(), "burst_frames")
                        os.makedirs(out_dir, exist_ok=True)
                        for i in range(self.burst_size):
                            f = cam.GetNextImage()
                            if not f.IsIncomplete():
                                burst.append(f.GetNDArray().copy())
                            f.Release()
                            self.progress.emit(int(100 * (i+1) / self.burst_size))
                        for i, arr in enumerate(burst):
                            fn = os.path.join(out_dir, f"frame_{i:03d}.png")
                            cv2.imwrite(fn, arr)
                        # process burst into panorama
                        strips = [flatten_strip(f, top, bot, self.height) for f in burst]
                        if self.no_stitch:
                            pano = cv2.hconcat(strips)
                        else:
                            pano = stitch_strips_with_flow(strips)
                        # cam.EndAcquisition()
                        # emit and hold until next event
                        self.finished.emit(pano)
                        # user_out_val.SetValue(False) 

                finally:
                    cam.EndAcquisition()
                    cam.DeInit()
                    system.ReleaseInstance()

                return
            # ── end AUTO MODE ────────────────────────────────────────────────

            # ── FRAMES FOLDER / VIDEO ────────────────────────────────────────
            strips = []
            if self.input_type == "Frames Folder":
                files = sorted(glob.glob(os.path.join(self.in_path, "*.*")))
                imgs = [f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))]
                if not imgs:
                    raise RuntimeError("No images found in folder.")
                for i, fn in enumerate(imgs):
                    img = cv2.imread(fn)
                    if img is None:
                        continue
                    strips.append(flatten_strip(img, top, bot, self.height))
                    self.progress.emit(int(100*(i+1)/len(imgs)))
            else:  # Video File
                cap = cv2.VideoCapture(self.in_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                for i in range(total):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    strips.append(flatten_strip(frame, top, bot, self.height))
                    self.progress.emit(int(100*(i+1)/total))
                cap.release()
                if not strips:
                    raise RuntimeError("Couldn't read any frames.")

            if self.no_stitch:
                pano = cv2.hconcat(strips)
            else:
                pano = stitch_strips_with_flow(strips)

            self.finished.emit(pano)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stretch Dashboard")
        self.resize(900, 700)

        # ── Input type radios ──────────────────────────────────────────
        r1 = QRadioButton("Frames Folder")
        r2 = QRadioButton("Video File")
        r3 = QRadioButton("Auto Mode")
        r1.setChecked(True)
        self.rg = QButtonGroup(self)
        for r in (r1, r2, r3):
            self.rg.addButton(r)
        self.rg.buttonToggled.connect(self.on_input_type_changed)

        # Input path & browse
        self.inputEdit   = QLineEdit()
        self.btnBrowse   = QPushButton("Browse…")
        self.btnBrowse.clicked.connect(self.browse)

        # Points JSON
        self.ptsEdit     = QLineEdit()
        self.ptsEdit.setReadOnly(True)

        # Frame selector (for folder/video)
        self.frameFileEdit = QLineEdit()
        self.frameFileEdit.setPlaceholderText("Choose image in folder…")
        self.btnFrameBrowse = QPushButton("Browse Frame")
        self.btnFrameBrowse.clicked.connect(self.browse_frame)
        self.frameNumSpin   = QSpinBox()
        self.frameNumSpin.setRange(0, 1_000_000)

        # ROI / threshold / burst size (Auto Mode only)
        self.roiXSpin         = QSpinBox(); self.roiXSpin.setPrefix("X=");    self.roiXSpin.setRange(0,10000)
        self.roiYSpin         = QSpinBox(); self.roiYSpin.setPrefix("Y=");    self.roiYSpin.setRange(0,10000)
        self.roiWSpin         = QSpinBox(); self.roiWSpin.setPrefix("W=");    self.roiWSpin.setRange(1,10000)
        self.roiHSpin         = QSpinBox(); self.roiHSpin.setPrefix("H=");    self.roiHSpin.setRange(1,10000)
        self.thresholdSpin    = QSpinBox(); self.thresholdSpin.setPrefix("Δ=");self.thresholdSpin.setRange(0,255); self.thresholdSpin.setValue(25)
        self.captureCountSpin = QSpinBox(); self.captureCountSpin.setPrefix("N=");self.captureCountSpin.setRange(1,1000); self.captureCountSpin.setValue(100)

        # Other controls
        self.heightSpin  = QSpinBox(); self.heightSpin.setRange(50,2000); self.heightSpin.setValue(400)
        self.chkNoStitch = QCheckBox("No-Stitch (concat only)")
        btnPick = QPushButton("Select 6 Points");  btnPick.clicked.connect(self.pick_points)
        btnRun  = QPushButton("Run Stretch");      btnRun.clicked.connect(self.run_process)

        self.progress   = QProgressBar()
        self.progress.setVisible(False)
        self.imageLabel = QLabel("Result will appear here")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # ── Layout ─────────────────────────────────────────────────────────
        topLay = QHBoxLayout(); topLay.addWidget(r1); topLay.addWidget(r2); topLay.addWidget(r3)
        pathLay = QHBoxLayout(); pathLay.addWidget(self.inputEdit); pathLay.addWidget(self.btnBrowse)
        ptsLay = QHBoxLayout(); ptsLay.addWidget(QLabel("Points JSON:")); ptsLay.addWidget(self.ptsEdit)
        frameLay = QHBoxLayout()
        frameLay.addWidget(QLabel("Frame:")); frameLay.addWidget(self.frameFileEdit)
        frameLay.addWidget(self.btnFrameBrowse); frameLay.addWidget(QLabel("Frame #")); frameLay.addWidget(self.frameNumSpin)

        roiLay = QHBoxLayout()
        for w in (self.roiXSpin, self.roiYSpin, self.roiWSpin, self.roiHSpin,
                  self.thresholdSpin, self.captureCountSpin):
            roiLay.addWidget(w)

        ctrlLay = QHBoxLayout()
        ctrlLay.addWidget(QLabel("Height:")); ctrlLay.addWidget(self.heightSpin)
        ctrlLay.addWidget(self.chkNoStitch); ctrlLay.addWidget(btnPick); ctrlLay.addWidget(btnRun)

        mainLay = QVBoxLayout()
        mainLay.addLayout(topLay)
        mainLay.addLayout(pathLay)
        mainLay.addLayout(ptsLay)
        mainLay.addLayout(frameLay)
        mainLay.addLayout(roiLay)
        mainLay.addLayout(ctrlLay)
        mainLay.addWidget(self.progress)
        mainLay.addWidget(self.imageLabel)

        container = QWidget()
        container.setLayout(mainLay)
        self.setCentralWidget(container)

        self.on_input_type_changed()  # initialize

    @property
    def input_type(self):
        if self.rg.buttons()[2].isChecked():
            return "Auto Mode"
        return "Frames Folder" if self.rg.buttons()[0].isChecked() else "Video File"

    def on_input_type_changed(self, *_):
        is_auto = (self.input_type == "Auto Mode")
        # disable browse/frame in auto
        for w in (self.inputEdit, self.btnBrowse,
                  self.frameFileEdit, self.btnFrameBrowse,
                  self.frameNumSpin):
            w.setEnabled(not is_auto)
        # show only ROI in auto
        for w in (self.roiXSpin, self.roiYSpin, self.roiWSpin, self.roiHSpin,
                  self.thresholdSpin, self.captureCountSpin):
            w.setVisible(is_auto)
        # update points path whenever mode changes
        self.update_pts_path()

    def update_pts_path(self):
        path = self.inputEdit.text().strip()
        if not path:
            self.ptsEdit.clear()
            return
        if self.input_type == "Frames Folder":
            pts = os.path.join(path, "points.json")
        else:
            d, fn = os.path.split(path)
            base, _ = os.path.splitext(fn)
            pts = os.path.join(d, f"{base}_points.json")
        self.ptsEdit.setText(pts)

    def browse(self):
        if self.input_type == "Frames Folder":
            d = QFileDialog.getExistingDirectory(self, "Choose Frames Folder")
            if d: self.inputEdit.setText(d)
        else:
            f, _ = QFileDialog.getOpenFileName(self, "Choose Video File",
                                               filter="Videos (*.mp4 *.avi *.mov)")
            if f: self.inputEdit.setText(f)
        self.update_pts_path()

    def browse_frame(self):
        base = self.inputEdit.text().strip()
        if not os.path.isdir(base):
            QMessageBox.warning(self, "Error", "Select a valid frames folder first.")
            return
        fn, _ = QFileDialog.getOpenFileName(self, "Select Frame Image", base,
                                            filter="Images (*.jpg *.jpeg *.png)")
        if fn:
            self.frameFileEdit.setText(fn)

    def pick_points(self):
        path, pts = self.inputEdit.text().strip(), self.ptsEdit.text().strip()
        if not path or not pts:
            QMessageBox.warning(self, "Error", "Choose input & points path first.")
            return
        try:
            if self.input_type == "Frames Folder":
                fn = self.frameFileEdit.text().strip() or next(
                    i for i in sorted(glob.glob(os.path.join(path,"*.*")))
                    if i.lower().endswith((".jpg",".jpeg",".png"))
                )
                select_points(fn, pts)
            else:
                idx = self.frameNumSpin.value()
                cap = cv2.VideoCapture(path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise RuntimeError(f"Couldn’t grab frame {idx}.")
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                tmp.close()
                select_points(tmp.name, pts)
                os.unlink(tmp.name)
            QMessageBox.information(self, "Saved", f"points.json →\n{pts}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_process(self):
        path, pts = self.inputEdit.text().strip(), self.ptsEdit.text().strip()
        if self.input_type != "Auto Mode":
            if not os.path.exists(path) or not os.path.isfile(pts):
                QMessageBox.warning(self, "Error", "Ensure input & points.json exist.")
                return

        self.progress.setVisible(True)
        self.progress.setValue(0)

        args = [
            self.input_type,
            path,
            pts,
            self.heightSpin.value(),
            self.chkNoStitch.isChecked()
        ]
        if self.input_type == "Auto Mode":
            args += [
                (self.roiXSpin.value(), self.roiYSpin.value(),
                 self.roiWSpin.value(), self.roiHSpin.value()),
                self.thresholdSpin.value(),
                self.captureCountSpin.value()
            ]

        self.thread = ProcessThread(*args)
        self.thread.progress.connect(self.progress.setValue)
        self.thread.finished.connect(self.on_done)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_done(self, pano):
        h, w, ch = pano.shape
        img = QImage(pano.data, w, h, ch*w, QImage.Format_BGR888)
        self.imageLabel.setPixmap(
            QPixmap.fromImage(img)
                  .scaled(self.imageLabel.size(),
                          Qt.KeepAspectRatio,
                          Qt.SmoothTransformation)
        )
        self.progress.setVisible(False)

        # save latest panorama
        if self.input_type == "Frames Folder":
            out = os.path.join(self.inputEdit.text(), "label_panorama.jpg")
        elif self.input_type == "Auto Mode":
            out = os.path.join(os.getcwd(), "auto_panorama.jpg")
        else:
            d, fn = os.path.split(self.inputEdit.text())
            base,_ = os.path.splitext(fn)
            out = os.path.join(d, f"{base}_panorama.jpg")
        cv2.imwrite(out, pano)
        # no exit—UI stays open and will update on next change

    def on_error(self, msg):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
    def on_done(self, pano):
        h, w, ch = pano.shape
        img = QImage(pano.data, w, h, ch*w, QImage.Format_BGR888)
        self.imageLabel.setPixmap(QPixmap.fromImage(img)
                                  .scaled(self.imageLabel.size(),
                                          Qt.KeepAspectRatio,
                                          Qt.SmoothTransformation))
        self.progress.setVisible(False)

        # save output
        if self.input_type == "Frames Folder":
            out = os.path.join(self.inputEdit.text(), "label_panorama.jpg")
        else:
            d, fn = os.path.split(self.inputEdit.text())
            base,_ = os.path.splitext(fn)
            out = os.path.join(d, f"{base}_panorama.jpg")
        cv2.imwrite(out, pano)
        QMessageBox.information(self, "Done", f"Saved panorama →\n{out}")

    def on_error(self, msg):
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", msg)

if __name__=="__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())