import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file
from PIL import Image
import numpy as np

# ==============================================================================
# 1. EXPLICIT RESNET-18 ARCHITECTURE
# No black boxes. We build the architecture identical to PyTorch's/HuggingFace's 
# ResNet-18 so we can load the checkpoint cleanly without Transformers.
# ==============================================================================
class ResNetConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, activation=True):
        super().__init__()
        self.convolution = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.normalization = nn.BatchNorm2d(out_c)
        self.has_activation = activation

    def forward(self, x):
        x = self.normalization(self.convolution(x))
        return F.relu(x) if self.has_activation else x

class ResNetShortCut(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.convolution = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_c)
        
    def forward(self, x):
        return self.normalization(self.convolution(x))

class ResNetBasicLayer(nn.Module):
    def __init__(self, in_c, out_c, stride, use_shortcut=False):
        super().__init__()
        self.layer = nn.ModuleList([
            ResNetConvLayer(in_c, out_c, 3, stride=stride, padding=1, activation=True),
            ResNetConvLayer(out_c, out_c, 3, stride=1, padding=1, activation=False)
        ])
        self.shortcut = ResNetShortCut(in_c, out_c, stride) if use_shortcut else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.layer[0](x)
        out = self.layer[1](out)
        return F.relu(out + residual)

class ResNetStage(nn.Module):
    def __init__(self, in_c, out_c, stride, use_shortcut):
        super().__init__()
        self.layers = nn.ModuleList([
            ResNetBasicLayer(in_c, out_c, stride, use_shortcut=use_shortcut),
            ResNetBasicLayer(out_c, out_c, 1, use_shortcut=False)
        ])
        
    def forward(self, x):
        x = self.layers[0](x)
        return self.layers[1](x)

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            ResNetStage(64, 64, 1, False),
            ResNetStage(64, 128, 2, True),
            ResNetStage(128, 256, 2, True),
            ResNetStage(256, 512, 2, True)
        ])
        
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = ResNetConvLayer(3, 64, kernel_size=7, stride=2, padding=3, activation=True)
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.pooler(self.embedder(x))

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Matches the `resnet.embedder` and `resnet.encoder` prefixes in the safetensors exactly
        self.resnet = nn.ModuleDict({
            "embedder": ResNetEmbedder(),
            "encoder": ResNetEncoder()
        })
        # nn.Sequential names its components "0" (Flattern), "1" (Linear). 
        # Making the Linear layer map cleanly to `classifier.1.weight`
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet["embedder"](x)
        x = self.resnet["encoder"](x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.classifier(x)

# ==============================================================================
# 2. EXPLICIT IMAGE PREPROCESSING
# Removing the magic from AutoImageProcessor
# ==============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    # 1. Resize shortest edge to 256
    aspect_ratio = image.width / image.height
    if image.width < image.height:
        new_w, new_h = 256, int(round(256 / aspect_ratio))
    else:
        new_w, new_h = int(round(256 * aspect_ratio)), 256
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 2. Center crop to 224x224
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    # 3. Convert to numpy and scale to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 4. Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # 5. Switch from (H, W, C) to (C, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))

    # 6. Convert to tensor and add batch dimension (1, C, H, W)
    return torch.tensor(img_array).unsqueeze(0)


def main():
    home = os.path.expanduser("~")
    dataset_path = os.path.join(home, "dataset/cats-image")
    model_path = os.path.join(home, "models/resnet-18")

    print("1. Loading Image from dataset...")
    dataset = load_dataset(dataset_path)
    # The dataset dict loading returns PIL Images natively
    image = dataset["test"]["image"][0]

    print("2. Preprocessing Image Manually...")
    input_tensor = preprocess_image(image)

    print("3. Instantiating Explicit ResNet-18 Model...")
    model = ResNet18(num_classes=1000)
    
    # Load directly from safetensors checkpoint
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_file(weights_path)
    
    # Due to naming parity, strict mapping loads all params gracefully
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("4. Running Inference...")
    with torch.no_grad():
        logits = model(input_tensor)
    
    predicted_idx = logits.argmax(-1).item()

    print("5. Mapping Output to Label...")
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    label = config["id2label"][str(predicted_idx)]
    print(f"\n✅ Expected Output: {label}")

if __name__ == "__main__":
    main()
