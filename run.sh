bazel run //resnet-18 -- \
  --model=$HOME/models/resnet-18/model.safetensors \
  --config=$HOME/models/resnet-18/config.json \
  --image=$HOME/dataset/cats-image/cats_image.jpeg
