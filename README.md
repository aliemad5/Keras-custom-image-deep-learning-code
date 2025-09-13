
Train a custom CNN classifier with Keras/TensorFlow.

Used for YOLO+Keras object detection model:https://github.com/aliemad5/YOLO-Keras-object-detection-model/blob/main/README.md

Dataset: CIFAR-100 

Author: Ali Emad Elsamanoudy

Email:ali.elsamanoudy623@gmail.com

## License
Copyright (c) 2025 Ali Emad Elsamanoudy
[MIT License](./LICENSE) — **CREDIT REQUIRED. DO NOT IGNORE.**

## Requirements
All dependencies for this project are listed in [Requirements0.txt](Requirements0.txt).
To install them, run the following command in your terminal:
```bash
pip install -r Requirements0.txt
```
## Imports
```python
import tensorflow as tf
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
```

# Load Dataset (CIFAR-100)

```python
print("[INFO] Loading CIFAR-100 dataset...")
(x_train, y_train), _ = tf.keras.datasets.cifar100.load_data(label_mode="fine")
```
# Normalize & resize
```python
x_train = x_train.astype("float32") / 255.0
x_train = tf.image.resize(x_train, [512, 512])
y_train = y_train.squeeze()

num_classes = 100
print(f"[INFO] CIFAR-100 loaded: {x_train.shape[0]} images, {num_classes} classes")
```

## Build Model

```python
model = Sequential([
    Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(512, 512, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (4, 4), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

print("[INFO] Model compiled.")
```

# Train

```python
print("[INFO] Starting training...")
model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2
)
```

# Save Model

```python
model.save("mykeras.h5")
print("[INFO] Model saved as mykeras.h5")
```
