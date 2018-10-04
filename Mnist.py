from keras.datasets import mnist
# import cv2
import tensorflow as tf
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# img = cv2.imread("one.png")

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)

x_train=x_train.astype("float32")
x_test=x_test.astype("float32")

x_train = x_train/255
x_test=x_test/255

print("training data shape is {}".format(x_train))
print("testing data shape is {}".format(x_test))

from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D

model=Sequential()
model.add(Conv2D(32,kernel_size=(5,5),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(5,5),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x=x_train,y=y_train,epochs=10)
test_error_rate=model.evaluate(x_test,y_test,verbose=0)
print("The mean squared error (MSE) for the data set is :{}".format(test_error_rate))
model.save("trained_model.h5")