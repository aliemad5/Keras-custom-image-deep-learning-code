
# Train a Custom CNN Classifier with Keras/TensorFlow

Used for [YOLO + Keras Object Detection Model](https://github.com/aliemad5/YOLO-Keras-object-detection-model/blob/main/README.md)

**Dataset:** Caltech-101  
**Author:** Ali Emad Elsamanoudy  
**Email:** ali.elsamanoudy623@gmail.com  

---

## License
Copyright (c) 2025 Ali Emad Elsamanoudy  
[MIT License](./LICENSE) — **Credit REQUIRED. Do NOT ignore.**

---

## Requirements
All dependencies for this project are listed in [Requirements.txt](Requirements.txt).  
To install them, run the following command in your terminal:

```bash
pip install -r Requirements.txt
```
## Imports
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
```

## Load Dataset (Caltech-101)

```python
dataset, info = tfds.load("caltech101", with_info=True, as_supervised=True)
train_ds = dataset["train"]

```
## Normalize & resize
```python
x_train, y_train = [], []

for img, label in tfds.as_numpy(train_ds):
    img = tf.image.resize(img, [512, 512]) / 255.0
    x_train.append(img)
    y_train.append(label)

x_train = np.array(x_train, dtype="float32")
y_train = np.array(y_train, dtype="int32")

num_classes = len(np.unique(y_train))
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

```

## Train

```python

model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=20,
    validation_split=0.2
)
```

## Save Model

```python
model.save("mykeras.h5")
print("[INFO] Model saved as mykeras.h5")
```
