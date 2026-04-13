import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


from model import build_model

MODEL_PATH = "../model/sign_model.pth"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ──────────────────────────────
ckpt    = torch.load(MODEL_PATH, map_location=DEVICE)
classes = ckpt["classes"]
model   = build_model(len(classes)).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ── Transform (same as val) ─────────────────
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Skin mask (same as preprocess) ──────────
def get_hand_crop(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,  20, 50],  dtype=np.uint8)
    upper1 = np.array([25, 180, 255], dtype=np.uint8)
    lower2 = np.array([155, 20, 50],  dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1),
                          cv2.inRange(hsv, lower2, upper2))

    h_img, w_img = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h_img+2, w_img+2), np.uint8), (0,0), 255)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(flood))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # ignore tiny detections (noise)
    if w * h < 5000:
        return None, None

    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    crop = frame[y1:y2, x1:x2]
    box  = (x1, y1, x2, y2)
    return crop, box

# ── Predict ─────────────────────────────────
def predict(crop_bgr):
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    inp = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(inp), dim=1)[0]
    top3  = probs.topk(3)
    return [(classes[i], probs[i].item()) for i in top3.indices]

# ── Main loop ───────────────────────────────
cap = cv2.VideoCapture(1)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop, box = get_hand_crop(frame)

    if box is not None:
        x1, y1, x2, y2 = box

        # draw box around hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        results = predict(crop)
        best_label, best_conf = results[0]

        # big prediction label
        cv2.putText(frame, f"{best_label}  {best_conf*100:.1f}%",
                    (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # top 3 predictions in corner
        for i, (label, conf) in enumerate(results):
            cv2.putText(frame, f"{i+1}. {label}: {conf*100:.1f}%",
                        (10, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Spanish Sign Language", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()