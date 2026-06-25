"""
Zone 2 + Zone 3 — CNN Models
Backbone: EfficientNet-B0 (faster than DenseNet-121, similar accuracy)
EfficientNet-B0 is 5x faster on CPU than DenseNet-121.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_image(img_array):
    if img_array.dtype != np.uint8:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return TRANSFORM(Image.fromarray(img_array)).unsqueeze(0)

def build_efficientnet(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

class LungModel(nn.Module):
    CLASSES = ['Normal', 'Infection']
    def __init__(self):
        super().__init__()
        self.net = build_efficientnet(len(self.CLASSES))
    def forward(self, x): return self.net(x)
    def predict(self, img_tensor, device='cpu'):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(img_tensor.to(device)), dim=1)[0].cpu().numpy()
        idx = probs.argmax()
        return self.CLASSES[idx], float(probs[idx]), probs

class BoneModel(nn.Module):
    CLASSES = ['Normal', 'Fracture']
    def __init__(self):
        super().__init__()
        self.net = build_efficientnet(len(self.CLASSES))
    def forward(self, x): return self.net(x)
    def predict(self, img_tensor, device='cpu'):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(img_tensor.to(device)), dim=1)[0].cpu().numpy()
        idx = probs.argmax()
        return self.CLASSES[idx], float(probs[idx]), probs

class BrainModel(nn.Module):
    CLASSES = ['Normal', 'Tumor']
    def __init__(self):
        super().__init__()
        self.net = build_efficientnet(len(self.CLASSES))
    def forward(self, x): return self.net(x)
    def predict(self, img_tensor, device='cpu'):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(img_tensor.to(device)), dim=1)[0].cpu().numpy()
        idx = probs.argmax()
        return self.CLASSES[idx], float(probs[idx]), probs

class ModalityRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_efficientnet(3)
        self.classes = ['chest', 'bone', 'brain']
    def forward(self, x): return self.backbone(x)
    def route(self, img_tensor, device='cpu'):
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(img_tensor.to(device)), dim=1)[0]
            idx   = probs.argmax().item()
        return self.classes[idx], probs.cpu().numpy()