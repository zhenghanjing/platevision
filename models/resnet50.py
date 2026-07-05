"""ResNet-50 classifier fine-tuned on Food-101."""

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from data.food101 import build_transforms
from models.checkpoint import load_checkpoint
from models.classifier import ClassificationResult, FoodClassifier

# Parameter-name prefix of the replaced head, used by freeze_all_except() during
# phase 1 (frozen-backbone) training.
HEAD_PREFIX = "fc."


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with ImageNet weights and its final FC layer replaced for `num_classes`."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet50Classifier(FoodClassifier):
    def load(self) -> None:
        checkpoint = load_checkpoint(self.weights_path, map_location=self.device)
        self.class_names: list[str] = checkpoint["class_names"]
        self.model = build_resnet50(num_classes=len(self.class_names), pretrained=False)
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
