# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:04:09 2020
Last update Monday Jan 10 12:11 pm 2022

该文件定义了Colangelo和Lee（2021）研究中用于模拟和实证应用的神经网络模型。所有神经网络均使用PyTorch包实现。
我们首先定义一个"NeuralNet"类，用于创建符合DDMLCT类要求的torch神经网络。
具体而言，DDMLCT要求模型必须同时具有.fit和.predict方法，因此NeuralNet类帮助我们实现这一要求。
我们还定义了NeuralNetk作为基础类，用于构建K神经网络，该网络在损失函数中采用了Colangelo和Lee（2021）中指定的额外核函数。
随后，我们为模拟和实证应用中使用的4个模型分别定义了NeuralNet的子类。
若需创建可与DDMLCT类兼容的新神经网络模型，只需按照如下方式定义NeuralNet或NeuralNetk的子类：
定义网络层结构（如下方示例所示），指定损失函数（criterion）、优化器（optimizer）和训练轮数（epochs）即可。

"""

import torch
import torch.nn as nn
import gc
import time
import logging
torch.set_default_tensor_type('torch.DoubleTensor')

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20,1))
        self.layer_1.add_module("R1", nn.SELU())
        
        
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.01)
        self.epochs = 100

    def weight_reset(self,m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()   
            
    def forward(self, x):
        for i in range(1,len(vars(self)['_modules'])):
            x = getattr(self,('layer_'+str(i)))(x)

        return x
    
    def fit(self,x,y):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y),1)

        for t in range(self.epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self(x)
            #logging.warning(t)
            # Compute and print loss
            loss = self.criterion(y_pred, y)
            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return self
            
    def predict(self,x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:,0]
def my_loss(output, target,K):
    loss = torch.mean(((1/2)*(output - target)**2)*K)
    return loss
    
class NeuralNetk(torch.nn.Module):
    def __init__(self):
        super(NeuralNetk, self).__init__()
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(20,1))
        self.layer_1.add_module("R1", nn.SELU())
        
        
        self.criterion = torch.nn.MSELoss()
        print(self.criterion)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.epochs = 300

    def weight_reset(self,m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()   
            
    def forward(self, x):
        for i in range(1,len(vars(self)['_modules'])):
            x = getattr(self,('layer_'+str(i)))(x)


        return x
    
    def fit(self,x,y,K):
        self.apply(self.weight_reset)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).reshape(len(y),1)
        K = torch.from_numpy(K)
        K2 = torch.sqrt(K).reshape(len(y),1)
        K2y = torch.mul(y,K2)
        for t in range(self.epochs):
            # 向前传球： 通过将 x 传递给模型来计算预测 y
            y_pred = self(x)
            #print(y_pred)

            #logging.warning(t)
            #K2y_pred = torch.mul(y_pred,K2)
            # Compute and print loss
            #loss = my_loss(y_pred,y,K)
            loss = self.criterion(torch.mul(y_pred,K2), K2y)


            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            #with torch.no_grad():
            loss.backward()
            self.optimizer.step()
            #del K2y_pred
            #loss = loss.detach()
            #gc.collect()
        return self
    def predict(self,x):
        x = torch.from_numpy(x)
        pred = self(x).detach().numpy()
        return pred[:,0]
    
class NeuralNet1_n10000(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet1_n10000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,self.k))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(self.k,self.k))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(self.k,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs
        
class NeuralNet1_n1000(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet1_n1000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(self.k,self.k))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs        
        
class NeuralNet1_emp_app(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet1_emp_app, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,25))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(self.k,self.k))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(25,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs       
        
class NeuralNet1k_n1000(NeuralNetk):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=100):
        super(NeuralNet1k_n1000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(100,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        
        self.layer_2 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs

class NeuralNet1k_n10000(NeuralNetk):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=100):
        super(NeuralNet1k_n10000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,100))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(100,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        
        self.layer_2 = torch.nn.Linear(100,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs

class NeuralNet1k_emp_app(NeuralNetk):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=100):
        super(NeuralNet1k_emp_app, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,25))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(100,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        
        self.layer_2 = torch.nn.Linear(25,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs

class NeuralNet2_n10000(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet2_n10000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,100))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(10,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(100,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs

class NeuralNet2_n1000(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet2_n1000, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,10))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(10,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(10,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs      

class NeuralNet2_emp_app(NeuralNet):
    def __init__(self,k,lr=0.01,momentum=0.5,weight_decay=0.2,epochs=300):
        super(NeuralNet2_emp_app, self).__init__()
        
        self.k = k
        
        self.layer_1 = nn.Sequential()
        self.layer_1.add_module("L1", nn.Linear(self.k,25))
        self.layer_1.add_module("R1", nn.ReLU())
        
        # self.layer_2 = nn.Sequential()
        # self.layer_2.add_module("L2", nn.Linear(10,10))
        # self.layer_2.add_module("R2", nn.ReLU())
        
        # self.layer_3 = nn.Sequential()
        # self.layer_3.add_module("L3", nn.Linear(10,10))
        # self.layer_3.add_module("R3", nn.ReLU())
        
        # self.layer_4 = nn.Sequential()
        # self.layer_4.add_module("L4", nn.Linear(10,10))
        # self.layer_4.add_module("R4", nn.ReLU())
        
        self.layer_2 = torch.nn.Linear(25,1)       

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,momentum=momentum,weight_decay=weight_decay)
        self.epochs = epochs 