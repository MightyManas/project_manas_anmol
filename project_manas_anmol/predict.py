# predict.py

import torch
from PIL import Image
import torchvision.transforms as transforms
from config import resize_x, resize_y, device


# Load class names from checkpoints/classes.txt
with open("checkpoints/classes.txt") as f:
    CLASSES = [line.strip() for line in f if line.strip()]


def predict_images(model, image_paths):

    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
    ])

    model.eval()
    model.to(device)

    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        preds = torch.argmax(outputs, dim=1)

    return [CLASSES[i] for i in preds.cpu().tolist()]
