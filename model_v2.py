import torch.nn as nn
import torchvision.models as models

class EfficientNetB2(nn.Module):

    def __init__(self):
        super().__init__()
        model = models.efficientnet_b2(weights="IMAGENET1K_V1")
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(1408, 3)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)