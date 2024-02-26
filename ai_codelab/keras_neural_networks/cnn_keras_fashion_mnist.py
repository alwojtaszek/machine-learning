import keras
import numpy as np
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from keras.src.utils import to_categorical
from keras.callbacks import EarlyStopping

from ai_codelab.utils.plot_helpers import plot_mismatched_labels

"""
Fashion MNIST CNN Implementation

This script implements a Convolutional Neural Network (CNN) using Keras for classifying images 
from the Fashion MNIST dataset.

Label Descriptions:
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

Architecture:
- Convolutional layer with 32 filters, kernel size (5, 5), and ReLU activation
- MaxPooling layer with pool size (2, 2)
- Convolutional layer with 64 filters, kernel size (3, 3), and ReLU activation
- MaxPooling layer with pool size (2, 2)
- Flatten layer
- Dense (fully connected) layer with 10 units and softmax activation

Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy

"""

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=10, activation='softmax'))

optimizer = keras.optimizers.Adam()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=20, callbacks=[early_stop])

pred = model.predict(x_test)

y_pred_label = [np.argmax(prediction) for prediction in pred]
y_pred_probability = [np.max(prediction) for prediction in pred]
mismatched_label_prediction = [(expected, actual) for expected, actual in zip(y_pred_label, y_test)
                               if expected != actual]

print(f'Number of test data: {len(y_test)}')
print(f'Number of mismatched predictions: {len(mismatched_label_prediction)}')
plot_mismatched_labels(x_test, y_pred_label, y_pred_probability, y_test)
plt.grid(False)
plt.tight_layout()
plt.show()
