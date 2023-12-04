# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 08:12:17 2023

@author: Hasan Emre
"""

#%% import library

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img

import matplotlib.pyplot as plt
from glob import glob  # kac tane class in oldugunu ogrenmek icin

#%%  dataset tanimlama ve yolunu belirtme daha sonrasında bir resim yazdirma
train_path = "fruits-360/Training/"
test_path = "fruits-360/Test/"

img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

#%% shape

x = img_to_array(img)
print(x.shape)

# training icerisindeki butun resimleri bir değiskene atadik daha sonrasında kac tane class numarasi var onu yazdirdik
className = glob(train_path + "/*")
numberOfClass = len(className)
print("Number of class: ", numberOfClass)


#%%  CNN Model

model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3), input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,kernel_size = (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,kernel_size = (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output
model.add(Activation("softmax"))

model.compile(loss= "categorical_crossentropy", optimizer = "rmsprop", metrics = ["accuracy"])

batch_size = 32

#%% Data Generation  -   Train Test Split

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range= 0.3, # bir resmi belirli bir acida saga yada sola cevirecek
    horizontal_flip= True,  # burda resmi 90 derecelik aci ile saga ya da sola yatiriyor
    zoom_range=0.3, # resmi yakinlastirip uzaklastiriyor
    )

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical"
    )

test_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical"
    )


history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=1600 // batch_size, # her bir epoch da kac adim atmasini hesaplıyor yani 1600 / 32 = 50 kere donuyor
    epochs=100,
    validation_data=test_generator,
    validation_steps = 800 // batch_size
    )

#%% model save

model.save_weights("Fruit.h5")

#%% model evaluation
print(history.history.keys())
plt.plot(history.history["loss"], label = "Train Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure() #◙ iki ayrı tabloda olusmalarini sagliyor

plt.plot(history.history["accuracy"], label = "Train accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation accuracy")
plt.legend()
plt.show()

#%% save history  
# yaptigimiz degisiklikleri kaydederek bir daha ki sefere kolaylik olmasi saglaniyor
import json
with open("cnn_fruit_history.json","w") as f:
    json.dump(history.history, f)


#%% load history  
# kaydetdegimiz yeri burada cagirarak bu sekilde kullanabiliyoruz
import json
import codecs
with codecs.open("cnn_fruit_history.json", "r", encoding="utf-8") as f:
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




