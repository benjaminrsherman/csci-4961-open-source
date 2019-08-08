from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (
    test_images,
    test_labels,
) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


train_images = train_images / 255.0

test_images = test_images / 255.0

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=5)

from PIL import Image

shoe_im = Image.open("shoe.png")
purse_im = Image.open("purse.png")
shirt_im = Image.open("shirt.png")

shoe = np.array(shoe_im)
purse = np.array(purse_im)
shirt = np.array(shirt_im)

test_images = np.array([shoe, purse, shirt])
test_images = test_images / 255

predictions = model.predict(test_images)

for prediction in predictions:
    predicted_label = np.argmax(prediction)
    print(f"{prediction} {predicted_label} {class_names[predicted_label]}")
