# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:09:56 2023

@author: Hasan Emre
"""

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Activation, Flatten, BatchNormalization
from keras_preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob 



#%% load and preprocess

def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.values
    np.random.shuffle(data)
    x = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
    y = data[:, 0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))
    
    return x, y

train_data_path = "mnist_train.csv"
test_data_path = "mnist_test.csv"
x_train, y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)


#%% visiualize

index = 8
vis = x_train.reshape(60000,28,28)
plt.imshow(vis[index,:,:])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index]))

#%% CNN
numberOfClass = y_train.shape[1]

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 128, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units= numberOfClass))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

# Train
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=25, batch_size= 4000)

#%% 
model.save_weights("cnn_mnist_model.h5")

#%%  Evulation

print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(hist.history["accuracy"], label = "Train accuracy")
plt.plot(hist.history["val_accuracy"], label = "Validation accuracy")
plt.legend()
plt.show()

#%% save history
import json
with open("cnn_mnist_hist.json", "w") as f:
    json.dump(hist.history, f)

#%% load history
import codecs
with codecs.open("cnn_mnist_hist.json","r", encoding="utf-8") as f:
    h = json.loads(f.read())
    
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()

plt.plot(h["accuracy"], label = "Train accuracy")
plt.plot(h["val_accuracy"], label = "Validation accuracy")
plt.legend()
plt.show()












