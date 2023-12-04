# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 12:35:43 2023

@author: Hasan Emre
"""

from keras.layers import Dense, Dropout, Input, ReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = mnist.load_data() # dataseti yukluyoruz

x_train = (x_train.astype(np.float32)-127.5)/127.5  # tum degerleri -1 ile 1 arasina sikistiriyoruz

print(x_train.shape) # (60000, 28 ,28)


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
print(x_train.shape)  #` (60000, 784)

#%%
plt.imshow(x_test[10])

#%%  create generator  (uretici ag (kalpazan))

# kalpazan her seferinde fake resimler olusturur ve dedektife gönderir. dedektif orjinal resimden ayirt edemeyene kadar kalpazan 
# resim yapmaya devam eder. burada ise her seferinde backpropagation ile weights leri günceller 
def create_generator():
    
    generator = Sequential()
    generator.add(Dense(units = 512, input_dim = 100))
    generator.add(ReLU())
    
    generator.add(Dense(units = 512))
    generator.add(ReLU())
 
    generator.add(Dense(units = 1024))
    generator.add(ReLU())
 
    generator.add(Dense(units = 784, activation = "tanh"))
    
    generator.compile(loss = "binary_crossentropy", 
                      optimizer = Adam(lr = 0.0001, beta_1=0.5))
    
    return generator

g = create_generator()
g.summary()
 
#%% create discriminator (ayirt edici ag (Dedektif))

# kalpazandan gelen resimleri dedektif orjinal resimden ayırt etmek icin burada discriminator egitiliyor

def create_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units = 1024, input_dim = 784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))

    discriminator.add(Dense(units = 512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units = 256))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units = 1,activation="sigmoid"))
    
    discriminator.compile(loss = "binary_crossentropy", 
                          optimizer = Adam(lr = 0.0001, beta_1= 0.5))
    return discriminator

d = create_discriminator()
d.summary()


#%%  GANs 

def create_gan(discriminator,generator):
    
    discriminator.trainable = False # burada dedektif train edilemeyecek yani backpropagation olmayacagi icin kendini gelistiremeyecek 
    # sayet dedektif her seferinde kendini gelistirirse kalpazanin her resminin fake oldugunu anlar ve kalpazan isini yapamaz
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan .compile(loss="binary_crossentropy", 
                 optimizer = "adam")
    
    return gan

gan = create_gan(d, g)
gan.summary()

#%% train

epochs = 50
batch_size = 256

for i in range(epochs):
    for _ in range(batch_size):
        
        noise = np.random.normal(0,1, [batch_size, 100])
        
        generated_images = g.predict(noise)
        
        image_batch = x_train[np.random.randint(low=0, high = x_train.shape[0], size = batch_size)]

        x = np.concatenate([image_batch, generated_images])
        
        y_dis = np.zeros(batch_size*2)
        y_dis[:batch_size] = 1
        
        d.trainable = True
        d.train_on_batch(x,y_dis)
        
        
        noise = np.random.normal(0,1,[batch_size,100])
        
        y_gen = np.ones(batch_size)
        
        d.trainable = False
        
        gan.train_on_batch(noise, y_gen)
    
    print("epochs: ", i)

#%% save model

g.save_weights("gans_model_weights_h5")

#%% visualize 

noise = np.random.normal(loc = 0, scale = 1, size = [100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.imshow(generated_images[99], interpolation="nearest")
plt.axis("off")
plt.show()







