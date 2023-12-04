# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:53:03 2023

@author: Hasan Emre
"""

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import json, codecs
import warnings
warnings.filterwarnings("ignore")

#%%
# burada dataseti her zamanki gibi değilde sadece x_train ve x_test kismini kullanacagimiz icin y li yerlere altan cizgi koyuyoruz ve kullanmiyoruz
(x_train, _ ),(x_test , _ ) = fashion_mnist.load_data()

x_train = x_train.astype("float32")/ 255.0   # x_train ve x_test tipini float32 ye cevirdik ve normalize ettik
x_test = x_test.astype("float32")/ 255.0

# x_train.shape  (60000, 28 , 28)
# x_train.shape[1:][0]   (28)

x_train = x_train.reshape((len(x_train),x_train.shape[1:][0] * x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test),x_test.shape[1:][0] * x_test.shape[1:][1]))

#%% visualize image

plt.imshow(x_train[1510].reshape(28,28))
plt.axis("off")
plt.show()

#%% 
# 784 e 1 lik bir vektorumuz bulunmaktadir bu da autoencoders un  input  girisi
input_img = Input(shape = (784,)) 

# encoded diye bir degisken tanimladik ve bu da inputtan sonraki hidden layerlar
encoded = Dense(64,activation="relu")(input_img)

encoded = Dense(64,activation="relu")(encoded)

encoded = Dense(32,activation="relu")(encoded)

# Dense mizi ayarladiktan sonraki paranteze bi rönce ki islemdeki degiskeni yazarak birbirlerine bagliyoruz
encoded = Dense(16, activation="relu")(encoded)

# en sonki encoded ile en derine indigimizi varsayarak artik birlestirme islemine yani decoded kismini yaziyoruz
decoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(decoded)

decoded = Dense(64, activation="relu")(decoded)

# en son olarak yapacagimiz output icin inputtaki shape ile ayni olmak zorunda ondan dolayi 784 olarak aldik
output_img = Dense(784,activation="sigmoid")(decoded)

# modelimizi burada en bastan sona topluyoruz
autoencoder = Model(input_img, output_img)

autoencoder.compile(optimizer = "rmsprop", loss = "binary_crossentropy")

hist = autoencoder.fit(x_train,x_train, # burada input ve output ayni olmasi icin iki kere x_train yazdik
                       epochs=500, 
                       batch_size = 64, 
                       shuffle = True,
                       validation_data=(x_train,x_train))


#%% save weights
autoencoder.save_weights("Autoencoders_fashionMNIST_weights.h5")

#%% evaluation (degerlendirme)
print(hist.history.keys())

plt.plot(hist.history["loss"], label= "train loss")
plt.plot(hist.history["val_loss"], label = "validation loss")
plt.legend()
plt.show()

#%% save history

with open("Autoencoders_fashionMNIST_history.json","w") as f:
    json.dump(hist.history, f)
    
#%%  load history

with codecs.open("Autoencoders_fashionMNIST_history.json") as f:
    n = json.loads(f.read())
    
plt.plot(n["loss"], label= "train loss")
plt.plot(n["val_loss"], label = "validation loss")
plt.legend()
plt.show()
  
 
#%%  modelimizi test ediyoruz

encoder = Model(input_img, encoded)
encoded_img = encoder.predict(x_test)

plt.imshow(x_test[1500].reshape(28,28))  # bu kisim orjinal olan resmi yazdiracak ve 
# daha sonra asagida ise autoencoder modeline girdikten sonraki resmi yazdiracak
plt.axis("off")
plt.show()

plt.figure()
# autoencoder modeline girdikten sonraki resim 
plt.imshow(encoded_img[1500].reshape(4,4))
plt.axis("off")
plt.show()

#%%  orjinal ve autoencoder resimlerini karsilastiriyoruz
decoded_imgs = autoencoder.predict(x_test)

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()





