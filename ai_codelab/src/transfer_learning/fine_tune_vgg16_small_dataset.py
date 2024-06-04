import os

import keras.datasets.cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.applications import VGG16
from keras.layers import Dense
from keras.utils import to_categorical

from ai_codelab.src.utils.plot_helpers import plot_mismatched_labels

"""
Utilizing the VGG16 convolutional neural network architecture for image classification
on the CIFAR-10 dataset using the Keras library.
"""

current_dir = os.getcwd()

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

img_size = x_test.shape[1]

vgg16 = VGG16(include_top=False, pooling='avg')
for layer in vgg16.layers[:-1]:
    layer.trainable = False

n_classes = 10
top_model = vgg16.output
ff_layer = Dense(n_classes, activation='softmax')(top_model)

model = Model(inputs=vgg16.input, outputs=ff_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat)
y_pred = np.argmax(model.predict(x_test), axis=-1)
y_pred_probability = np.max(model.predict(x_test), axis=-1)
mismatched_label_prediction = [(expected, actual) for expected, actual in zip(y_pred, y_test)
                               if expected != actual]

print(f'Number of test data: {len(y_test)}')
print(f'Number of mismatched predictions: {len(mismatched_label_prediction)}')
plot_mismatched_labels(x_test, y_pred, y_pred_probability, y_test)
plt.grid(False)
plt.tight_layout()
plt.show()
