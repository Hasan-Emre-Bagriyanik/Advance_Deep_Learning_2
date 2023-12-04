# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:04:43 2023

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

x_train_negative_tensor = torch.from_numpy(train_negative_array[:42000,:]) #  arraylerin boyutlarına tensor denir pytorch da 
print("x_train_negative_tensor: ",x_train_negative_tensor.size())

y_train_negative_tensor = torch.zeros(42000, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
print("y_train_negative_tensor: ",y_train_negative_tensor.size())

#%%

# read train positive  
train_positive_path = r"C:\Users\Hasan Emre\Documents\Yapay Zeka Python\Python Yapay Zeka Çalışmaları\Advance Deep Learning\Deep Residual Networks (ResNets)\LSIFIR\Classification\Train\pos" 
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path, num_train_positive_img) # belirdeigimiz yolda fonksiyona yazarak resimleri tek tek aliyoruz

x_train_positive_tensor = torch.from_numpy(train_positive_array[:10000,:]) #  arraylerin boyutlarına tensor denir pytorch da 
print("x_train_positive_tensor: ",x_train_positive_tensor.size())

y_train_positive_tensor = torch.zeros(10000, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
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

x_test_negative_tensor = torch.from_numpy(test_negative_array[:18056,:]) #  arraylerin boyutlarına tensor denir pytorch da....  burada train disinda bir de sinirlama getiriyoruz 
print("x_test_negative_tensor: ",x_test_negative_tensor.size())

y_test_negative_tensor = torch.zeros(18056, dtype = torch.long) # sifirlardan olusan birtek boyutlu dizimiz var
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



import torch.utils.data       
train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size= batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size= batch_size, shuffle = False)


#%% 
def conv3x3(in_planes, out_planes, strides = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride=strides, padding=1,  bias=False)

def conv1x1(in_planes, out_planes, strides = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride=strides,  bias=False)


class basicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride = 1, downsample = None):
        super(basicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.9)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.relu(out)
        return out
        

        
class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes = num_classes):
        super(ResNet,self).__init__()
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride = 2, padding= 1)
        self.layers1 = self._make_layer(block, 64, layers[0] , stride = 1)
        self.layers2 = self._make_layer(block, 128, layers[1] , stride = 2)
        self.layers3 = self._make_layer(block, 256, layers[2] , stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256*block.expansion, num_classes)
        
        # model calisamazsa yani ogrenemezse
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # Conv2d layerindaysam weightlerimi nasil gunceleyecem 
                nn.init.kaiming_normal_(m.weight, mode = "fan_out", nonlinearity="relu") # kaiming_normal = sifira yakin degerleri weightlerimize atiyor
            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1) # weightleri  1 e esitle demek
                nn.init.constant_(m.bias,0)   # biaslari 0 e esitle demek
                
                
        
    
    def _make_layer(self, block, planes, blocks, stride = 1): # inplanes = inputs, planes = outputs, block = basicBlock, blocks = kaç tane basicblock oldugu
        """
            block: Bu parametre, temel blok olarak kullanılacak bir sınıfı temsil eder."basicBlock" Bu temel blok, katmandaki her bir tekrarlanan bloğun yapı taşıdır.
            planes: Bu parametre, katmandaki çıkış kanallarının sayısını belirtir. Kanallar, katmanın çıktı tensöründeki derinlik boyutunu temsil eder.
            blocks: Bu parametre, katmanda tekrarlanacak olan temel blokların sayısını belirtir.
            stride: Bu parametre, katmanın adımını belirtir. Varsayılan değeri 1'dir.
        """
        self.downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            self.downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride), 
                    nn.BatchNorm2d(planes*block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, self.downsample))
        self.inplanes = planes*block.expansion
        
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
        """
            Kod, önceki katmanın çıktı boyutunu kontrol etmek ve katmanın başında veya adımda boyut değişikliği gerekiyorsa 
            "downsample" adı verilen bir geçit oluşturmak için gerekli kontrolleri gerçekleştirir. Bu, katmanın çıktı boyutunu,
            sonraki katmana aktarabilmek için uygun şekilde ayarlamayı sağlar.
        """
    



    def forward(self,x):
        
        # sira ile cnn modeli gibi olusturyoruz 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x


model = ResNet(basicBlock, [2,2,2])

# model = ResNet(basicBlock, [2,2,2]).to(device) GPU icin

#%%  loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)



#%% train 

loss_list = []
train_acc = []
test_acc = []
use_gpu = True

total_step = len(trainloader)

for epoch in range(num_epochs):
    for i , (images, labels) in enumerate(trainloader):
        images = images. view(batch_size,1,64,32)
        images = images.float()
        
        # gpu
        if use_gpu:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)
                
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        # backward amd optimization
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print("epoch : {}  {}/{}".format(epoch,i,total_step))
            
            
    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.sizea(0)
            correct  += (predicted == labels).sum().item()
    
    print("Accuracy train %d %% "%(100*correct/total))
    train_acc.append(100*correct/total)


    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(batch_size,1,64,32)
            images = images.float()
            
            # gpu
            if use_gpu:
                if torch.cuda.is_available():
                    images, labels = images.to(device), labels.to(device)
                    
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.sizea(0)
            correct  += (predicted == labels).sum().item()
    
    print("Accuracy test %d %% "%(100*correct/total))
    test_acc.append(100*correct/total)
    
    loss_list.append(loss.item())
    
    












