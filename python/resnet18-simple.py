import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> torch.Tensor:
    print("1. Preprocessing image...")
    aspect_ratio = image.width / image.height
    if image.width < image.height:
        new_w, new_h = 256, int(round(256 / aspect_ratio))
    else:
        new_w, new_h = int(round(256 * aspect_ratio)), 256
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))

    img_array = np.array(image, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_array = np.transpose(img_array, (2, 0, 1))

    inputs = torch.tensor(img_array).unsqueeze(0)
    return inputs


def main():
    home = os.path.expanduser("~")
    dataset_path = home + "/dataset/cats-image"
    model_path = home + "/models/resnet-18"

    dataset = load_dataset(dataset_path)
    image = dataset["test"]["image"][0]

    inputs = preprocess_image(image)

    print("2. Loading model...")
    model = AutoModelForImageClassification.from_pretrained(model_path)

    print("3. Running explicit inference steps...")
    with torch.no_grad():
        outputs = model.resnet(inputs)
        pooled_output = outputs.pooler_output
        flattened_output = pooled_output.flatten(1)
        logits = model.classifier(flattened_output)

    predicted_label = logits.argmax(-1).item()
    print("\nPredicted label:", model.config.id2label[predicted_label])


if __name__ == "__main__":
    main()
