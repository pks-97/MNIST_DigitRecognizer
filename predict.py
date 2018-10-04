import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D
import tensorflow as tf
img = cv2.imread("two.jpg")
[h,w,c] = img.shape
# resized = cv2.resize(img, (28,28))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (28,28))
im=np.array(resized)
x_test=im.reshape(1,28,28,1)
print("the height is {} and the width is {}".format(h,w))
# cv2.imshow("resized",resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
model=keras.models.load_model("trained_model.h5")
predictions = model.predict(x_test)
p = predictions[0] 
c = 0
for i in p:

	if i==1:
		print(c)
		break
	c=c+1			

# print(predictions[0][4])

