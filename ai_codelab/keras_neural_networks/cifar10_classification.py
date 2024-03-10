# To fine-tuning yet
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.datasets import cifar10
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, RandomTranslation
from keras.src.utils import to_categorical

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = x_train.reshape(-1, image_size, image_size, 3) / 255.0
x_test = x_test.reshape(-1, image_size, image_size, 3) / 255.0

model = Sequential()
model.add(RandomTranslation(0.1, 0.1, input_shape=(32, 32, 3)))
model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(kernel_size=(3, 3), filters=64, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(kernel_size=(3, 3), filters=128, activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.build()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, callbacks=[early_stop])
loss, accuracy = model.evaluate(x_test, y_test)

print(loss, accuracy)
# 0.9523851275444031 0.7687000036239624
