# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:11:19 2023

@author: Hasan Emre
"""

from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import cv2
import numpy as np

#%% 

(x_train, y_train),(x_test, y_test)  = cifar10.load_data()  # cifar10 datasetini keras kutuphanesinden hazır bir sekilde aldik ve bu sekilde yukleme yapiyoruz
print("x_train shape: ", x_train.shape)

print("train sample: ", x_train.shape[0])

numberOfClass = 10 # toplamda  tane class var 

y_train = to_categorical(y_train, numberOfClass)  # burada y_train ve y_test i kategorilerine ayiriyoruz 
y_test = to_categorical(y_test, numberOfClass)

input_shape = x_train.shape[1:]


#%% visualize

plt.imshow(x_train[5511].astype(np.uint8))
plt.axis("off")
plt.show()

#%% increase dimesion (resimlerin boyularini buyutme)

def resize_img (img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

x_train = resize_img(x_train)
x_test = resize_img(x_test)
print("increased dim x_train: ", x_train.shape)

plt.figure()
plt.imshow(x_train[5511].astype(np.uint8))
plt.axis("off")
plt.show()

#%%  vgg19 

VGG19 = VGG19(include_top=False, weights="imagenet", input_shape= (48,48,3))

print(VGG19.summary())

vgg_layer_list = VGG19.layers
print(vgg_layer_list)

model = Sequential()
for layer in vgg_layer_list:
    model.add(layer)
    
print(model.summary())
    
# trainlerimi train etme 
for layer in model.layers:
    layer.trainable = False

# fully con layers

model.add(Flatten())()
model.add(Dense(128))
model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary())


model.compile(loss = "categorical_crossentropy", 
              optimizer = "rmsprop",
              metrics= ["accuracy"])
hist = model.fit(x_train, y_train, validation_split= 0.2, epochs = 5, batch_size = 1000)


#%% save weights

model.save_weights("Deneme_weights.h5")
#%% visualize

plt.plot(hist.history["loss"], label = "train loss")
plt.plot(hist.history["val_loss"], label = "validation loss")
plt.legend()
plt.figure()
plt.plot(hist.history["accuracy"], label = "train accuracy")
plt.plot(hist.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()

#%%  save history
import json,codecs
with open("transfer_learning_vgg19_cfar10.json","w") as f:
    json.dump(hist.history, f)
    
#%% load history
with codecs.open("transfer_learning_vgg19_cfar10.json","r", encoding="utf-8") as f:
    n = json.loads(f.read())
    
plt.plot(n["loss"], label = "train loss")
plt.plot(n["val_loss"], label = "validation loss")
plt.legend()
plt.figure()
plt.plot(n["accuracy"], label = "train accuracy")
plt.plot(n["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()




