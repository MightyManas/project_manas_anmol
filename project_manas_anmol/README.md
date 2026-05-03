# Ninjutsu-Vision 🤙
**Naruto Hand Sign Recognition with Augmented Reality Jutsu Overlay**

A real-time computer vision system that recognizes the 12 fundamental Naruto hand seals and triggers animated jutsu overlays when a correct sign sequence is performed.

---

## Project Structure

```
project_manas_anmol/
│
├── checkpoints/
│   ├── final_weights.pth       # Best model weights (saved during training)
│   ├── best_model.pth          # Alternate checkpoint
│   └── classes.txt             # Class label list (13 classes)
│
├── data/
│   ├── train/                  # Training images, one folder per class
│   └── test/                   # 10 sample images per class (raw, unresized)
│
├── assets/
│   ├── fireball.png
│   └── lightning_cutter_(chidori).png
│
├── training/
│   ├── run_train.py            # Standalone training entry point
│   └── eval.py                 # Evaluation script
│
├── utils/
│   ├── jutsu_dataset.json
│   └── save_classes.py
│
├── dataset.py                  # NarutoDataset class + naruto_loader
├── model.py                    # NarutoModel (MobileNetV2 fine-tuned)
├── train.py                    # Training loop (train_model function)
├── predict.py                  # Inference on image paths (predict_images)
├── interface.py                # Standardized imports for grading
├── config.py                   # All hyperparameters and path config
├── video_inference.py          # Live webcam demo with AR overlay
├── jutsu_sequences.json        # Jutsu name → hand sign sequence mapping
├── demo.mp4                    # Demo video showing the AR overlay in action
└── req.txt                     # Dependencies
```

---

## Setup

**Install dependencies:**
```bash
pip install -r req.txt
```

**Dependencies:** `torch`, `torchvision`, `opencv-python`, `mediapipe`, `Pillow`, `numpy`

---

## Demo

A pre-recorded demonstration video is included as `demo.mp4` in the root directory. It shows the live webcam pipeline in action — hand detection, sign classification, sequence building, and the AR jutsu overlay triggering on a completed sequence.

---

## How to Run

### Train the model
```bash
python training/run_train.py
```
Trains from `data/train` and `data/test`, saves the best checkpoint to `checkpoints/final_weights.pth`.

### Evaluate on test set
```bash
python training/eval.py
```

### Run inference on images
```python
from model import NarutoModel
from predict import predict_images

model = NarutoModel()
model.load_state_dict(torch.load("checkpoints/final_weights.pth"))

preds = predict_images(model, ["data/bird/img01.jpg", "data/tiger/img01.jpg"])
```

### Live webcam demo (AR overlay)
```bash
python video_inference.py
```
> Press `ESC` to quit. Change the camera index in `video_inference.py` (`cv2.VideoCapture(2)`) if your webcam isn't detected.

---

## Model

- **Architecture:** MobileNetV2 (pretrained on ImageNet, fine-tuned)
- **Classes:** 13 — `bird, boar, dog, dragon, hare, horse, monkey, ox, ram, rat, snake, tiger, zero`
- **Input size:** 224 × 224 × 3
- **Framework:** PyTorch

---

## How the AR Overlay Works

1. **Hand detection** — MediaPipe detects hand landmarks and crops a bounding box around both hands.
2. **Classification** — The cropped region is passed through NarutoModel. A prediction is only accepted if confidence > 0.85 and the margin between top-2 predictions > 0.2.
3. **Stability filter** — A rolling buffer of 7 frames requires 5 consistent predictions before a sign is registered.
4. **Sequence matching** — Detected signs are appended to a sequence buffer. When the tail of the buffer matches a jutsu's required sequence (from `jutsu_sequences.json`), the jutsu triggers.
5. **Overlay** — A PNG asset is alpha-composited onto the frame for 2 seconds.

### Jutsu Sequences

| Jutsu | Sequence |
|---|---|
| Fire Release: Great Fireball | Snake → Ram → Monkey → Boar → Horse → Tiger |
| Lightning Cutter (Chidori) | Ox → Hare → Monkey |
| Impure World Reincarnation | Tiger → Snake → Dog |

### `jutsu_sequences.json` structure

This file drives all jutsu detection. Each entry needs a `name`, `sequence` (ordered list of sign names in lowercase), and an `asset` path pointing to the PNG overlay:

```json
{
  "jutsu": [
    {
      "name": "Lightning Cutter (Chidori)",
      "sequence": ["ox", "hare", "monkey"],
      "length": 3,
      "asset": "assets/lightning_cutter_(chidori).png"
    }
  ]
}
```

To add a new jutsu, append a new entry with the correct sign sequence and drop the corresponding PNG into `assets/`. The overlay image should have a transparent background (RGBA) for clean compositing.

---

## Configuration

All hyperparameters live in `config.py`:

| Variable | Value | Description |
|---|---|---|
| `batchsize` | 32 | Training batch size |
| `epochs` | 6 | Number of training epochs |
| `learning_rate` | 1e-3 | Adam optimizer LR |
| `resize_x / resize_y` | 224 / 224 | Input image dimensions |
| `input_channels` | 3 | RGB |
| `num_classes` | 13 | Number of hand sign classes |
| `train_dir` | `data/train` | Training data path |
| `val_dir` | `data/test` | Validation data path |

---

## Interface (for grading)

`interface.py` exposes standardized names used by the grading program:

```python
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader
from interface import the_batch_size, total_epochs
```

| Alias | Points to |
|---|---|
| `TheModel` | `NarutoModel` in `model.py` |
| `the_trainer` | `train_model` in `train.py` |
| `the_predictor` | `predict_images` in `predict.py` |
| `TheDataset` | `NarutoDataset` in `dataset.py` |
| `the_dataloader` | `naruto_loader` in `dataset.py` |

The `train_model` function signature matches the spec:
```python
def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
```

---

## Dataset

- **Source:** [Naruto Hand Signs Dataset — Kaggle](https://www.kaggle.com/)
- **Size:** ~5,000 labeled images across 12 sign classes
- **Augmentation:** RandomHorizontalFlip, RandomRotation(10°), ColorJitter applied during training
- The `data/` directory contains 10 raw sample images per class for grading purposes
