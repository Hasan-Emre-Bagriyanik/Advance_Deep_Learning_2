# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 13:18:10 2023

@author: Hasan Emre
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob



train_path = "fruits-360/Training/"  # yollari belirttik 
test_path = "fruits-360/Test/"


img = load_img(train_path + "Apple Golden 1/0_100.jpg")  # herhangi bir resimi göstermek icin bir dosya yolunda resmi yazdirdik
plt.imshow(img)
plt.axes("off")
plt.show()

#%% 
x = img_to_array(img)
print(x.shape)


numberOfClass = len(glob(train_path + "/*") ) # tüm meyve cesitlerinin kac tane oldugunun sayisini burada aliyoruz
#%%

vgg = VGG16()

print(vgg.summary())  # vgg icindeki layerlari gosterir

#%%

print(type(vgg))  # keras modeli oldugunu ekrana yazdirir

vgg_layer_list = vgg.layers
print(vgg_layer_list)  # summary ile aynı seyi yazdirir

#%%

model = Sequential()
for i in range(len(vgg_layer_list) - 1):
    model.add(vgg_layer_list[i])   # vgg icindeki 22 tane layer i tek tek modelimize ekliyoruz
    
    
print(model.summary())


#%%

for layers in model.layers:  # her bir layeri train edilsin mi edilmesin mi sorusuna train edilmeyerek devam edecek
    layers.trainable = False
    
    
model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary())

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])


#%% train

# 100 e 100 luk resimlerimizi ImageDataGenerator() ile resim boyutlarini 224 e 224 luk yapacaktir 
train_Data = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224))
test_Data = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224))

batch_size = 128

hist = model.fit_generator(train_Data,
                           steps_per_epoch=1600//batch_size,
                           epochs=10, 
                           validation_data=test_Data,
                           validation_steps=800//batch_size)

#%% save weights

model.save_weights("denem.h5")

#%% evaluation

print(hist.history.keys())

plt.plot(hist.history["loss"], label = "training loss")
plt.plot(hist.history["val_loss"], label = "validation loss")
plt.legend()
plt.figure()
plt.plot(hist.history["accuracy"], label = "training accuracy")
plt.plot(hist.history["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()

#%% save history
import json,codecs
with open("VGG16_Fruits_History.json","w") as f:
    json.dump(hist.history,f)
    

#%% load history
with codecs.open("VGG16_Fruits_History.json","r",encoding="utf-8") as f:
    n = json.loads(f.read())
    
plt.plot(n["loss"], label = "training loss")
plt.plot(n["val_loss"], label = "validation loss")
plt.legend()
plt.figure()
plt.plot(n["accuracy"], label = "training accuracy")
plt.plot(n["val_accuracy"], label = "validation accuracy")
plt.legend()
plt.show()
    



