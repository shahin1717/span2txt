import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = "../data/A/DSC00882.JPG"

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Skin mask
lower1 = np.array([0,  20, 50],  dtype=np.uint8)
upper1 = np.array([25, 180, 255], dtype=np.uint8)
lower2 = np.array([155, 20, 50],  dtype=np.uint8)
upper2 = np.array([180, 180, 255], dtype=np.uint8)

mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                      cv2.inRange(hsv, lower2, upper2))

h_img, w_img = mask.shape
flood = mask.copy()
cv2.floodFill(flood, np.zeros((h_img+2, w_img+2), np.uint8), (0,0), 255)
flood_inv = cv2.bitwise_not(flood)
mask = cv2.bitwise_or(mask, flood_inv)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

# Crop
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h  = cv2.boundingRect(max(contours, key=cv2.contourArea))

pad_x = int(w * 0.15)
pad_y = int(h * 0.15)
x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
x2, y2 = min(img_bgr.shape[1], x + w + pad_x), min(img_bgr.shape[0], y + h + pad_y)
cropped = img_rgb[y1:y2, x1:x2]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_rgb);          axes[0].set_title("Original")
axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Skin Mask")
axes[2].imshow(cropped);          axes[2].set_title("Cropped Hand")
for ax in axes: ax.axis("off")
plt.tight_layout()
plt.savefig("mask_test.png")
print("Saved → mask_test.png")