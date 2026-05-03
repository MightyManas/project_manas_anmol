# video_inference.py

import cv2
import torch
import mediapipe as mp
import torchvision.transforms as transforms
from PIL import Image
from collections import deque, Counter
import torch.nn.functional as F
import time
import json
import os

from model import NarutoModel
from config import resize_x, resize_y, device


# ------------------------
# Load model + classes
# ------------------------
model = NarutoModel()
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

with open("checkpoints/classes.txt") as f:
    classes = [line.strip() for line in f]


# ------------------------
# Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
])


# ------------------------
# Load Jutsu
# ------------------------
with open("jutsu_sequences.json") as f:
    JUTSU_LIST = json.load(f)["jutsu"]

# normalize sequences
for j in JUTSU_LIST:
    j["sequence"] = [s.lower() for s in j["sequence"]]


# ------------------------
# Load PNG assets
# ------------------------
ASSETS = {}
for jutsu in JUTSU_LIST:
    path = jutsu.get("asset", None)
    if path and os.path.exists(path):
        ASSETS[jutsu["name"]] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        ASSETS[jutsu["name"]] = None


# ------------------------
# Overlay function
# ------------------------
def overlay_png(frame, png, x, y, scale=1.0):
    if png is None:
        return frame

    h, w = png.shape[:2]
    png = cv2.resize(png, (int(w * scale), int(h * scale)))

    h, w = png.shape[:2]

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    if png.shape[2] == 4:
        alpha = png[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                alpha * png[:, :, c] +
                (1 - alpha) * frame[y:y+h, x:x+w, c]
            )
    else:
        frame[y:y+h, x:x+w] = png

    return frame


# ------------------------
# MediaPipe
# ------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ------------------------
# Parameters
# ------------------------
CONF_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.2
STABILITY_THRESHOLD = 5
COOLDOWN = 0.5
JUTSU_COOLDOWN = 2.0
JUTSU_DURATION = 2.0


# ------------------------
# State
# ------------------------
buffer = deque(maxlen=7)
sequence = []
last_label = None
prev_box = None

last_added_time = 0
last_jutsu_time = 0

current_jutsu = ""
current_asset = None
jutsu_start_time = 0


# ------------------------
# Camera (change index if needed)
# ------------------------

try : cap = cv2.VideoCapture(2)
else : cap = cv2.VideoCapture(0)

# ------------------------
# Main loop
# ------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        # -------- Combine both hands --------
        all_x, all_y = [], []
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                all_x.append(lm.x)
                all_y.append(lm.y)

        x_min = int(min(all_x) * w)
        x_max = int(max(all_x) * w)
        y_min = int(min(all_y) * h)
        y_max = int(max(all_y) * h)

        # padding
        pad = int(0.3 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        # square crop
        size = max(x_max - x_min, y_max - y_min)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        x_min = max(0, cx - size // 2)
        y_min = max(0, cy - size // 2)
        x_max = min(w, cx + size // 2)
        y_max = min(h, cy + size // 2)

        # smooth box
        if prev_box is not None:
            x_min = int(0.7 * prev_box[0] + 0.3 * x_min)
            y_min = int(0.7 * prev_box[1] + 0.3 * y_min)
            x_max = int(0.7 * prev_box[2] + 0.3 * x_max)
            y_max = int(0.7 * prev_box[3] + 0.3 * y_max)

        prev_box = (x_min, y_min, x_max, y_max)

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        cv2.imshow("hand_crop", crop)

        # -------- Inference --------
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            probs = F.softmax(output, dim=1)

            top2 = torch.topk(probs, 2, dim=1)
            conf1 = top2.values[0][0].item()
            conf2 = top2.values[0][1].item()
            pred = top2.indices[0][0].item()

        if conf1 < CONF_THRESHOLD or (conf1 - conf2) < MARGIN_THRESHOLD:
            continue

        label = classes[pred]

        # -------- Stability --------
        buffer.append(label)
        top_label, count = Counter(buffer).most_common(1)[0]

        if count < STABILITY_THRESHOLD:
            continue

        stable_label = top_label

        # -------- Sequence --------
        if stable_label != last_label and (time.time() - last_added_time > COOLDOWN):
            sequence.append(stable_label)
            last_label = stable_label
            last_added_time = time.time()

            if len(sequence) > 10:
                sequence.pop(0)

            print("Sequence:", sequence)

        # -------- Jutsu detection --------
        for jutsu in JUTSU_LIST:
            seq = jutsu["sequence"]

            if len(sequence) >= len(seq):
                if sequence[-len(seq):] == seq:

                    if time.time() - last_jutsu_time < JUTSU_COOLDOWN:
                        continue

                    current_jutsu = jutsu["name"]
                    current_asset = ASSETS.get(current_jutsu)
                    jutsu_start_time = time.time()
                    last_jutsu_time = time.time()

                    print(f"JUTSU ACTIVATED: {current_jutsu}")

                    sequence = []
                    last_label = None
                    break

        # draw box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{stable_label} ({conf1:.2f})",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # -------- Display jutsu --------
    if current_jutsu:
        if time.time() - jutsu_start_time < JUTSU_DURATION:

            cv2.putText(frame, current_jutsu, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if current_asset is not None:
                fx, fy = frame.shape[1] // 2, frame.shape[0] // 2
                frame = overlay_png(frame, current_asset, fx - 150, fy - 150, 0.5)

        else:
            current_jutsu = ""
            current_asset = None

    cv2.imshow("Ninjutsu Vision", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
