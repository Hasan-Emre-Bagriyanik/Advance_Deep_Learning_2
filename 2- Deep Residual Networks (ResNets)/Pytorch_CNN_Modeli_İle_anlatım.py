# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:11:17 2023

@author: Hasan Emre
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#%% device confic ekstra default CPU but CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#%%

def read_images(path, num_img):
    array= np.zeros([num_img, 64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img  # yolun icindeki resimleri tek tek aliyor
        img = Image.open(img_path, mode = "r")  # resimleri acip okuyoruz
        data = np.asarray(img, dtype = "uint8") # daha sonra resimleri unit8 formatına ceviriyoruz
        data = data.flatten() # burada ise resimleri flatten ederek duzlestiriyoruz
        array[i,:] = data # butun i. resimlere bu formati uyguluyoruz
        i += 1
    return array     
#%%
  
# read train negative  
train_negative_path = r"C:\Users\Hasan Emre\Documents\Yapay Zeka Python\Python Yapay Zeka Çalışmaları\Advance Deep Learning\Deep Residual Networks (ResNets)\LSIFIR\Classification\Train\neg" 
num_train_negative_img = 43390
train_negative_array = read_images(train_negative_path, num_train_negative_img) # belirdeigimiz yolda fonksiyona yazarak resimleri tek tek aliyoruz

x_train_negative_tensor = torch.from_numpy(train_negative_array) #  arraylerin boyutlarına tensor denir pytorch da 
print("x_train_negative_tensor: ",x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(num_train_negative_img, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
print("y_train_negative_tensor: ",y_train_negative_tensor.size())

#%%

# read train positive  
train_positive_path = r"C:\Users\Hasan Emre\Documents\Yapay Zeka Python\Python Yapay Zeka Çalışmaları\Advance Deep Learning\Deep Residual Networks (ResNets)\LSIFIR\Classification\Train\pos" 
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path, num_train_positive_img) # belirdeigimiz yolda fonksiyona yazarak resimleri tek tek aliyoruz

x_train_positive_tensor = torch.from_numpy(train_positive_array) #  arraylerin boyutlarına tensor denir pytorch da 
print("x_train_positive_tensor: ",x_train_positive_tensor.size())

y_train_positive_tensor = torch.zeros(num_train_positive_img, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
print("y_train_positive_tensor: ",y_train_positive_tensor.size())

#%% concat train

x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor),0) # torch ta concat (birlestirme) islemi cat() ile yapiliyor
y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor),0)
print("x_train :", x_train.size())
print("y_train :", y_train.size())



#%%  


# read test negative  22050
test_negative_path = r"C:\Users\Hasan Emre\Documents\Yapay Zeka Python\Python Yapay Zeka Çalışmaları\Advance Deep Learning\Deep Residual Networks (ResNets)\LSIFIR\Classification\Test\neg" 
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path, num_test_negative_img) # belirdeigimiz yolda fonksiyona yazarak resimleri tek tek aliyoruz

x_test_negative_tensor = torch.from_numpy(test_negative_array[:20855,:]) #  arraylerin boyutlarına tensor denir pytorch da....  burada train disinda bir de sinirlama getiriyoruz 
print("x_test_negative_tensor: ",x_test_negative_tensor.size())

y_test_negative_tensor = torch.zeros(20855, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
print("y_test_negative_tensor: ",y_test_negative_tensor.size())

#%%

# read test positive  5944 

test_positive_path = r"C:\Users\Hasan Emre\Documents\Yapay Zeka Python\Python Yapay Zeka Çalışmaları\Advance Deep Learning\Deep Residual Networks (ResNets)\LSIFIR\Classification\Test\pos" 
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path, num_test_positive_img) # belirdeigimiz yolda fonksiyona yazarak resimleri tek tek aliyoruz

x_test_positive_tensor = torch.from_numpy(test_positive_array) #  arraylerin boyutlarına tensor denir pytorch da 
print("x_test_positive_tensor: ",x_test_positive_tensor.size())

y_test_positive_tensor = torch.zeros(num_test_positive_img, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
print("y_test_positive_tensor: ",y_test_positive_tensor.size())

#%% concat test

x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor),0) # torch ta concat (birlestirme) islemi cat() ile yapiliyor
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor),0)
print("x_test :", x_test.size())
print("y_test :", y_test.size())


#%% vizualize

plt.imshow(x_train[48005,:].reshape(64,32), cmap = "gray")

#%% CNN

# Hyperparameter

num_epochs = 5000
num_classes = 2
batch_size = 8933
learning_rate = 0.00001


class Net(nn.Module): # Net diye bir class olusturduk ve bunu nn.Module üzerinden inheritance yaptik
    
    def __init__(self):
        super(Net,self).__init__() # inheritance yaptigimiz class i super ile iceriye aktariyoruz
        
        self.conv1 = nn.Conv2d(1,10,5) # normal cnn modleindeki gibi Conv2d metodunu burada tanimladik
        self.pool = nn.MaxPool2d(2,2)# MaxPooling yaptik 2,2 seklinde 
        self.conv2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5, 520)
        self.fc2 = nn.Linear(520, 130)
        self.fc3 = nn.Linear(130, num_classes)
        
        
    def forward(self,x):
         x = self.pool(F.relu(self.conv1(x))) # Conv1 u relu aktivasyon icinde sokuyoruz Bu F ise yukaridaki torch un nn.functional kismindan geliyor. 
         # Daha sonra  pooling kisminda giriyor ve tek satirla bir hidden layer olusturmus oluyoruz
         x = self.pool(F.relu(self.conv2(x)))
         
         x = x.view(-1,16*13*5) # keras ta flatten formati pytorch ta bu sekilde 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return x

import torch.utils.data       
train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size= batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle = False)

net = Net()
# net = Net().to(device) Gpu icin


#%%  Loss and optimizer

criterion = nn.CrossEntropyLoss()

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum=0.8)

#%% train a network

start = time.time()

train_acc = []
test_acc = []
loss_list = []

use_gpu = False  # GPU varsa True

for epochs in range(num_epochs):
    for i , data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs = inputs.view(batch_size, 1, 64, 32)  # 64x32 lik resimler ve 1 yani siyah beyaz renkler  # reshape 
        inputs = inputs.float()  # float a cevirdik
        
        
        # use gpu
        if use_gpu:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
                
        
        # zero gradient
        optimizer.zero_grad()
        
        # forward
        outputs = net(inputs)
        
        # loss
        loss = criterion(outputs,labels)
        
        # backward
        loss.backward()
        
        # update weights
        optimizer.step()
        
        
    # test
    correct = 0
    total = 0
    with torch.no_grad():# bu komut ile backward sona eriyor
        for data in testloader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64 ,32) # reshape
            images = images.float()
            
            # use gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = net(images)
            
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0) # ne kadar data var
            correct += (predicted == labels).sum().item()  # toplam variable
    
    acc1 = 100*correct/total
    print("Epochs: {}, Acuracy test: {}".format(epochs + 1,acc1))
    test_acc.append(acc1)
    

    # train
    correct = 0
    total = 0
    with torch.no_grad():# bu komut ile backward sona eriyor
        for data in trainloader:
            images, labels = data
            
            images = images.view(batch_size, 1, 64 ,32) # reshape
            images = images.float()
            
            # use gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = net(images)
            
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc2 = 100*correct/total
    print("Epochs: {}, Acuracy train: {}".format(epochs + 1,acc2))
    train_acc.append(acc2)        



end = time.time()

process_time = (end-start) / 60
print("Process time: ", process_time)














