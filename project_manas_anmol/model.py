# model.py

import torch.nn as nn
import torchvision.models as models
from config import num_classes


class NarutoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.mobilenet_v2(pretrained=True)

        # Replace classifier
        self.model.classifier[1] = nn.Linear(
            self.model.last_channel, num_classes
        )

    def forward(self, x):
        return self.model(x)
