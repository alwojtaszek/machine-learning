from keras import Sequential
from keras.src.datasets import mnist
from keras.src.layers import Dense
from keras.src.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
x_train = x_train.reshape(-1, image_size*image_size) / 255.0
x_test = x_test.reshape(-1, image_size*image_size) / 255.0

model = Sequential()
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)
loss, accuracy = model.evaluate(x_test, y_test)
print(loss, accuracy)
# 0.2715926170349121 0.9265999794006348
