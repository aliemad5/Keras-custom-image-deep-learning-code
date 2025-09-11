# Keras-custom-image-deep-learning-code
## Imports
import cv2
import tensorflow as tf
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing import image
import pandas as pd
import numpy as np

## Load Data
```python
# ylabels.csv contains numeric labels (0 → 299)
# imgpaths.csv contains image file paths
y_train = pd.read_csv("ylabels.csv")        # shape: (9000, 1)
x_images = pd.read_csv("imgpaths.csv")      # shape: (9000, 1)

x_train = []
for img_path in x_images["path"]:
    img = image.load_img(img_path, target_size=(512, 512))
    img = image.img_to_array(img)
    img = img / 255.0
    x_train.append(img)
```
# Convert to numpy arrays
```python
x_train = np.array(x_train)
y_train = np.array(y_train).squeeze()  
```
## Build Model
model = Sequential([
    Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=(512, 512, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (4, 4), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(300, activation="softmax")  # 300 classes
])

model.compile(optimizer="adam",
              loss=SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

## Train Model
model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)

## Save Model
model.save("mykeras.h5")

