# import cv2
# from pyzbar import pyzbar
# import pytesseract
# import re
# import json
# import numpy as np
# from matplotlib import pyplot as plt
# import easyocr
# from doctr.models import ocr_predictor
# from doctr.io import DocumentFile
# from ollama import chat

# # can you read this image and see if the batch number and exp date are placed properly left side to the barcode and centered? are they? (answer yes or no)
# def ollama_chat(model='gemma3:4b'):
#     """
#     Interact with Ollama chat model.
#     """
#     response = chat(model=model, messages=[{
#             "role": "user",
#             "content": "do you see any wrinkles or bubbles on the label? (answer yes or no)",
#             "images": ["C:\\Users\\Shanka\\OneDrive - Monash University\\BLQC\\thresh.jpg"],
#         }
#     ],)
#     return response['message']['content']

# def enhance_contrast_bgr(img):
#     """
#     Enhance contrast of a BGR image using CLAHE on the L channel in LAB color space.
#     """
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(50,50))
#     cl = clahe.apply(l)
#     merged = cv2.merge((cl, a, b))
#     cv2.imwrite('enhanced_image.jpg', cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))
#     return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# def extract_barcode_and_positions(image_path, output_json=None, output_image_path='barcode_annotated.jpg'):
#     """
#     Detects barcodes in the image (any orientation), extracts data and positions.
#     Enhances contrast prior to decoding.
#     Returns list of dicts: {'type','data','position':{x,y,w,h},'angle'}
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Cannot open {image_path}")

#     # Copy for annotation
#     annotated = img.copy()
#     h, w = annotated.shape[:2]
#     y0, y1 = int(0.5*h), int(0.9*h)    # e.g. middle 30% of the strip
#     roi = annotated[y0:y1, :]
#     # Enhance contrast (if available)
#     img_enhanced = enhance_contrast_bgr(roi)
#     gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
#     # _, thresh = cv2.threshold(gray, 140, 255,  cv2.THRESH_BINARY)
#     # 110
#     _, bw = cv2.threshold(
#         gray, 
#         110, 
#         255, 
#         cv2.THRESH_BINARY 
#     )
#     results = []
#     # Try multiple orientations
#     barcodes = pyzbar.decode(bw)
#     cv2.imwrite('barcode_detection.jpg', bw)
#     for barcode in barcodes:
#         x, y, w_box, h_box = barcode.rect
#         pos = {'x': x, 'y': y, 'w': w_box, 'h': h_box}
#         # Annotate
#         cv2.rectangle(
#             roi,
#             (x, y),
#             (x + w_box, y + h_box),
#             (0, 255, 0),
#             2
#         )
#         text = f"{barcode.type}: {barcode.data.decode('utf-8')}"
#         cv2.putText(
#             roi,
#             text,
#             (x, y - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 255, 0),
#             2
#         )

#         results.append({
#             'type': barcode.type,
#             'data': barcode.data.decode('utf-8'),
#             'position': pos
#         })


#     # Save results and annotated image
#     if output_json:
#         with open(output_json, 'w') as f:
#             json.dump(results, f, indent=2)

#     cv2.imwrite(output_image_path, annotated)
#     return results



# def extract_batch_and_expiry(image_path, output_json=None):
#     """
#     Uses OCR to find Batch No and Expiry Date in the image.
#     Enhances contrast for better OCR.
#     Returns dict: {'batch':{value,position}, 'expiry':{value,position}}
#     """
#     img = cv2.imread(image_path)
#     annotated = img.copy()
#     if img is None:
#         raise FileNotFoundError(f"Cannot open {image_path}")
#     # enhance contrast
#     img = enhance_contrast_bgr(img)
#     cv2.GaussianBlur(img, (3,3), 0, img)
#     cv2.imwrite('enhanced_for_ocr.jpg', img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # + cv2.THRESH_OTSU
#     # _, thresh = cv2.threshold(gray, 140, 255,  cv2.THRESH_BINARY)
#     thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,9,4)
#     # thresh = cv2.bitwise_not(thresh)
#     # cv2.GaussianBlur(thresh, (7,7), 0, thresh)
#     cv2.imwrite('thresh.jpg', thresh)
    
#     config = "--psm 11"
#     data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)
#     # data2 = easyocr.Reader(['en']).readtext(thresh, detail=1, paragraph=False)
#     # for detection in data2:
#     #     print(detection[1])
#     batch_info = None
#     # model = ocr_predictor(pretrained=True)
#     # doc = DocumentFile.from_images(image_path)
#     # result = model(doc)
#     # print(result)
#     expiry_info = None
#     n = len(data['text'])
#     print(data['text'])
#     for i in range(n):
#         word = data['text'][i].strip()
#         if not word:
#             continue
#         m1 = re.search(r'(?i)(12/2027|12-2027)', word)
#         if m1 and not expiry_info:
#             x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             expiry_info = {'expiry_info position': {'x':x,'y':y,'w':w_box,'h':h_box}}
#             cv2.rectangle(annotated, (x,y), (x+w_box, y+h_box), (0,255,0), 2)
#         m2 = re.search(r'(?i)24747', word)
#         if m2 and not batch_info:
#             x, y, w_box, h_box = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
#             batch_info = {'batch_info position': {'x':x,'y':y,'w':w_box,'h':h_box}}
#             cv2.rectangle(annotated, (x,y), (x+w_box, y+h_box), (0,255,0), 2)
#         if batch_info and expiry_info:
#             break
#     result = {'batch': batch_info, 'expiry': expiry_info}
#     if output_json:
#         with open(output_json, 'w') as f:
#             json.dump(result, f, indent=2)
#     cv2.imwrite('batch_and_exp_annotated.jpg', annotated)
#     return result

    
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser(description='Extract barcode, batch no, expiry from image')
#     parser.add_argument('image', help='Path to input image')
#     parser.add_argument('--json', help='Write results to JSON file')
#     args = parser.parse_args()

#     barcodes = extract_barcode_and_positions(args.image, args.json)
#     be = extract_batch_and_expiry(args.image, args.json)
#     img = cv2.imread(args.image)
#     annotated = img.copy()
#     if img is None:
#         raise FileNotFoundError(f"Cannot open {args.image}")

#     # img = enhance_contrast_bgr(img)

#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray,100,200)
#     lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
#     vertical_lines = []
#     for line in lines:
#         x1,y1,x2,y2 = line[0]
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#         ang = abs( np.degrees( np.arctan2((y2-y1),(x2-x1))) )
#         if abs(ang - 90) < 15:
#             vertical_lines.append((x1,y1,x2,y2))
            
#     # response_msg = ollama_chat()
#     # print("Ollama response:", response_msg)
    
#     # if not vertical_lines:
#     #     print("No vertical lines found")
#     # else:
#     #     # pick the one with smallest x (leftmost)
#     #     first = min(vertical_lines, key=lambda t: min(t[0], t[2]))
#     #     x1,y1,x2,y2 = first

#     #     # draw it on your annotated copy
#     #     cv2.line(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
#     #     print(f"Found first vertical line at x ≈ {min(x1,x2)}px")

#     #     # show result
#     #     cv2.imshow("First vertical edge", annotated)
#     #     cv2.imwrite('longest_vertical.jpg',annotated)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
#     # cv2.imwrite('houghlines5.jpg',img)
    
#     print('Barcodes:', barcodes)
#     print('Batch & Expiry:', be)
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

def detect_longest_line(
    img,
    roi_config_path="roi_config.json",
    roi_key="line_roi",
    angle_filter_deg=45,      # None => ignore angle; else keep only lines within ±this many degrees of horizontal
    canny_low=20,
    canny_high=150,
    hough_threshold=100,
    hough_min_line_len=5,
    hough_max_line_gap=100,
    draw=True,
    line_color=(255, 178, 0),
    line_thickness=3
):
    try:
        H, W = img.shape[:2]

        # Load ROI if present; else use full image
        # roi = {"x": 0, "y": 0, "w": W, "h": H}
        # if os.path.exists(roi_config_path):
        #     try:
        #         with open(roi_config_path, "r") as f:
        #             cfg = json.load(f)
        #             if roi_key in cfg:
        #                 roi = cfg[roi_key]
        #     except Exception as e:
        #         logging.warning(f"Failed reading ROI config ({roi_config_path}): {e}")

        # x, y, w, h = [int(roi[k]) for k in ["x", "y", "w", "h"]]
        # x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        # w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

        # roi_img = img[y:y+h, x:x+w]
        roi_img = img
        if roi_img.size == 0:
            logging.warning("Empty ROI for line detection; using full image.")
            roi_img = img
            x, y, w, h = 0, 0, W, H

        # Edges & lines
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.waitKey(1)  # Allow the window to update
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
                label = f"{length:.1f}px @ {angle_deg:.1f}° (tilt {abs(angle_deg):.1f}°)"
                tx = min(max(10, min(X1, X2)), W - 260)
                ty = min(max(30, min(Y1, Y2)), H - 10)
                cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        print(f"Longest line: {result['length_px']:.1f}px @ {result['angle_deg']:.1f}° (tilt {result['tilt_from_horizontal_deg']:.1f}°)")
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
            column = frame[:, mid-2:mid+3, :]
            column_buffer.append(column)
        cap.release()
        
        if not column_buffer:
            raise ValueError("No valid frames found in video.")
        
        return np.hstack(column_buffer)

    except Exception as e:
        logging.error(f"Column stitching failed: {e}")
        return np.array([])
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
    
video_path = "test_videos\\temp-08012025122714-0000.avi"    
stitched = extract_column_stitch(video_path)
if stitched.size == 0:
    raise ValueError("Failed to extract stitched image from video.")

enhanced = enhance_contrast(stitched)
detect_longest_line(enhanced)