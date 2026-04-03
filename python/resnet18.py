import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from datasets import load_dataset

home = os.path.expanduser("~")
dataset_path = home + "/dataset/cats-image"
model_path = home + "/models/resnet-18"

dataset = load_dataset(dataset_path)
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
