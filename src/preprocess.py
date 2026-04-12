import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


DATA_DIR    = "../data"
OUTPUT_DIR  = "../data_cleaned"

def get_skin_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,  20, 50],  dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([155, 20, 50],  dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))

    # fill holes
    h_img, w_img = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h_img+2, w_img+2), np.uint8), (0,0), 255)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(flood))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def crop_hand(img_bgr, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr  # fallback: return original

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_bgr.shape[1], x + w + pad_x)
    y2 = min(img_bgr.shape[0], y + h + pad_y)
    return img_bgr[y1:y2, x1:x2]

def process_all():
    classes = sorted(os.listdir(DATA_DIR))
    total_saved, total_failed = 0, 0

    for cls in classes:
        src_dir = os.path.join(DATA_DIR, cls)
        dst_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        files = [f for f in os.listdir(src_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]

        for fname in tqdm(files, desc=f"[{cls}]"):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)

            img_bgr = cv2.imread(src_path)
            if img_bgr is None:
                print(f"  [SKIP] cant read: {src_path}")
                total_failed += 1
                continue

            mask    = get_skin_mask(img_bgr)
            cropped = crop_hand(img_bgr, mask)
            cv2.imwrite(dst_path, cropped)
            total_saved += 1

        print(f"[{cls}] done — {len(files)} images")

    print(f"\nFinished: {total_saved} saved, {total_failed} failed")
    print(f"Output → {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all()