"""EfficientNet-B4 classifier fine-tuned on Food-101."""

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from data.food101 import build_transforms
from models.checkpoint import load_checkpoint
from models.classifier import ClassificationResult, FoodClassifier

# EfficientNet's classifier head is Sequential(Dropout, Linear) -- "classifier.1"
# is the Linear layer we replace and the only part trainable during phase 1.
HEAD_PREFIX = "classifier.1."


def build_efficientnet_b4(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B4 with ImageNet weights and its final Linear layer replaced for `num_classes`."""
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class EfficientNetB4Classifier(FoodClassifier):
    def load(self) -> None:
        checkpoint = load_checkpoint(self.weights_path, map_location=self.device)
        self.class_names: list[str] = checkpoint["class_names"]
        self.model = build_efficientnet_b4(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        # Older checkpoints predate the image_size field; 224 was the only size used then.
        self.transform = build_transforms("test", image_size=checkpoint.get("image_size", 224))

    def predict(self, image_crop: np.ndarray) -> ClassificationResult:
        if self.model is None:
            raise RuntimeError("Call load() before predict().")

        image = Image.fromarray(image_crop).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        class_id = int(torch.argmax(probs).item())

        return ClassificationResult(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=float(probs[class_id]),
        )
