# dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import resize_x, resize_y, batchsize


class NarutoDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img))
                self.labels.append(idx)

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((resize_x, resize_y)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize_x, resize_y)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


def naruto_loader(root_dir, train=True):
    dataset = NarutoDataset(root_dir, train=train)
    return DataLoader(dataset, batch_size=batchsize, shuffle=train)
