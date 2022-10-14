# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import math
from DataPro import dataset_generator,dataset_separator,testset_estimator

# dataset
class MyDataset(Dataset):
    def __init__(self, one_hot, label):
        self.one_hot = one_hot
        self.label = label
        self.size = one_hot.shape[0]

    def __getitem__(self, index):
        one_hot_vector = self.one_hot[index]
        label_value = self.label[index]
        return one_hot_vector, label_value

    def __len__(self):
        return self.size

# fully connected network
class NeuTomography(nn.Module):
    def __init__(self, in_dim, neu_numbers, out_dim):
        super(NeuTomography, self).__init__()
        self.in_dim = in_dim
        self.neu_numbers = neu_numbers
        self.out_dim = out_dim
        self.layer1 = nn.Sequential(nn.Linear(self.in_dim, self.neu_numbers), nn.Sigmoid())
        self.layer2 = nn.Sequential(nn.Linear(self.neu_numbers, self.neu_numbers), nn.Sigmoid())
        self.layer3 = nn.Sequential(nn.Linear(self.neu_numbers, self.out_dim, bias = False))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# setup
root_dir = './rocketfuel-weights'
AS = ['/3967', '/1221', '/3257']
file_name = ['/latencies.intra']

sampling_rate = [0.2, 0.25, 0.3]
batch_size = 1
epochs = 1000

# output
LA_print = np.zeros((3,10))
LR_min = np.ones((3,10))*50000000
LR = np.zeros((3,10))

AS_print = ['3967', '1221', '3257']
LA = ['BPR_Re']
sampling_rate_print = ['20%', '25%', '30%']
sampling_type_print = [' random ', ' monitor', '   PAT  ', '   PPO  ', 
                       'PPO+PAT1', 'PPO+PAT2']

# start

# for AS_idx in range(2, 3):  # AS3257
for AS_idx in range(2):  # AS3967 and AS1221

    print('Experiments on AS'+AS_print[AS_idx]+':')

    host_l,one_hot_l,label_l = dataset_generator(root_dir+AS[AS_idx]+file_name[0])
    
    for i in range(3):
        random_l = dataset_separator(host_l,one_hot_l,label_l,sampling_rate[i],'random')
        monitor_l = dataset_separator(host_l,one_hot_l,label_l,sampling_rate[i],'monitor')
        
        dataaug_one_hot = []
        dataaug_label = []
        
        pat_onehot, pat_label, ppo_onehot, ppo_label, \
        ppop1_onehot, ppop1_label, ppop2_onehot, ppop2_label, MAPERL = testset_estimator(
            monitor_l[0],monitor_l[1],monitor_l[2],monitor_l[3])
        dataaug_one_hot.append(pat_onehot)
        dataaug_label.append(pat_label)
        dataaug_one_hot.append(ppo_onehot)
        dataaug_label.append(ppo_label)
        dataaug_one_hot.append(ppop1_onehot)
        dataaug_label.append(ppop1_label)
        dataaug_one_hot.append(ppop2_onehot)
        dataaug_label.append(ppop2_label)
        
        LA_print = np.zeros((3,10))
        LR_min = np.ones((3,10))*50000000
        LR = np.zeros((3,10))
        
        for lr_c in range(10):
            learning_rate = (lr_c + 1) * 0.0001
            print('learning_rate:',learning_rate)
            
            # random
            trainset_random_l = MyDataset(random_l[0], random_l[1])
            testset_random_l = MyDataset(random_l[2], random_l[3])
            trainloader_random_l = DataLoader(trainset_random_l, 
                                              batch_size=batch_size, shuffle=True)
            testloader_random_l = DataLoader(testset_random_l, 
                                             batch_size=batch_size, shuffle=False)
            # monitor
            trainset_monitor_l = MyDataset(monitor_l[0], monitor_l[1])
            testset_monitor_l = MyDataset(monitor_l[2], monitor_l[3])
            trainloader_monitor_l = DataLoader(trainset_monitor_l, 
                                               batch_size=batch_size, shuffle=True)
            testloader_monitor_l = DataLoader(testset_monitor_l, 
                                              batch_size=batch_size, shuffle=False)
            
            in_dim = random_l[0].shape[1]
            neu_numbers = math.ceil(2.5*in_dim)
            out_dim = 1
                            
            trainloader = [trainloader_random_l, trainloader_monitor_l]
            testloader = [testloader_random_l, testloader_monitor_l]
            for j in range(2):
                model = NeuTomography(in_dim, neu_numbers, out_dim)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                if torch.cuda.is_available():
                    model = model.cuda()
                
                # train
                for epoch in range(epochs):
                    train_loss = []
                    for data in trainloader[j]:
                        nodepair,metric = data
                        nodepair = nodepair.float()
                        metric = metric.float()
                        nodepair = Variable(nodepair)
                        metric = Variable(metric)
                        if torch.cuda.is_available():
                            nodepair = nodepair.cuda()
                            metric = metric.cuda()
                        
                        out = model(nodepair)
                        loss = criterion(out, metric.reshape(metric.shape[0],1))
                        train_loss.append(loss.data.item())
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    with torch.no_grad():
                        test_loss = []
                        for data in testloader[j]:
                            nodepair,metric = data
                            nodepair = nodepair.float()
                            metric = metric.float()
                            nodepair = Variable(nodepair)
                            metric = Variable(metric)
                            if torch.cuda.is_available():
                                nodepair = nodepair.cuda()
                                metric = metric.cuda()
                            
                            out = model(nodepair)
                            loss = criterion(out, metric.reshape(metric.shape[0],1))
                            test_loss.append(loss.data.item())
                    # if (epoch+1)%250 == 0:
                        # print('epoch: {}, train loss: {}, test loss: {}'.format(epoch+1, np.mean(train_loss), np.mean(test_loss)))
                
                # test
                # model.eval()
                MAPE_part = 0
                with torch.no_grad():
                    test_loss = []
                    for data in testloader[j]:
                        nodepair,metric = data
                        nodepair = nodepair.float()
                        metric = metric.float()
                        nodepair = Variable(nodepair)
                        metric = Variable(metric)
                        if torch.cuda.is_available():
                            nodepair = nodepair.cuda()
                            metric = metric.cuda()
                        
                        out = model(nodepair)
                        loss = criterion(out, metric.reshape(metric.shape[0],1))
                        test_loss.append(loss.data.item())
                        MAPE_part += torch.sum(torch.abs((out-metric)/metric))
                    
                    # print('test loss: {}'.format(np.mean(test_loss)))
                MAPE = MAPE_part.cpu()*100/len(testloader[j])
                LA_print[i][j] = MAPE
            
            # PAT
            alpha = 0.15
            beta = 0.6
            iterations = 6
            MAX = 10 ** 10
            
            model = NeuTomography(in_dim, neu_numbers, out_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            if torch.cuda.is_available():
                model = model.cuda()
                
            for j in range(2,6):
                
                for itr in range(iterations):
                    index = random.sample(range(0,dataaug_one_hot[j-2].shape[0]),math.ceil(alpha*dataaug_one_hot[j-2].shape[0]))
                    trainset_label_temp = dataaug_label[j-2][index]
                    index_temp = [i for i in range(len(index)) if dataaug_label[j-2][index[i]] == MAX]
                    for i in range(len(index_temp)):
                        idx = random.randint(0,dataaug_one_hot[j-2].shape[0]-1)
                        while idx in index or dataaug_label[j-2][idx] == MAX:
                            idx = random.randint(0,dataaug_one_hot[j-2].shape[0]-1)
                        index.append(idx)
                    index = np.delete(index,index_temp)
                    
                    trainset_onehot = dataaug_one_hot[j-2][index,:]
                    trainset_label = dataaug_label[j-2][index]
                    trainset_onehot = np.concatenate((monitor_l[0],trainset_onehot),axis=0)
                    trainset_label = np.concatenate((monitor_l[1],trainset_label))
                    trainset_l = MyDataset(trainset_onehot, trainset_label)
                    trainloader_l = DataLoader(trainset_l, 
                                                      batch_size=batch_size, shuffle=True)
                    
                    testset_l = MyDataset(monitor_l[2], monitor_l[3])
                    testloader_l = DataLoader(testset_l, 
                                              batch_size=batch_size, shuffle=False)
                    
                    # train
                    for epoch in range(epochs):
                        train_loss = []
                        for data in trainloader_l:
                            nodepair,metric = data
                            nodepair = nodepair.float()
                            metric = metric.float()
                            nodepair = Variable(nodepair)
                            metric = Variable(metric)
                            if torch.cuda.is_available():
                                nodepair = nodepair.cuda()
                                metric = metric.cuda()
                            
                            out = model(nodepair)
                            loss = criterion(out, metric.reshape(metric.shape[0],1))
                            train_loss.append(loss.data.item())
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        
                        with torch.no_grad():
                            test_loss = []
                            for data in testloader_l:
                                nodepair,metric = data
                                nodepair = nodepair.float()
                                metric = metric.float()
                                nodepair = Variable(nodepair)
                                metric = Variable(metric)
                                if torch.cuda.is_available():
                                    nodepair = nodepair.cuda()
                                    metric = metric.cuda()
                                
                                out = model(nodepair)
                                loss = criterion(out, metric.reshape(metric.shape[0],1))
                                test_loss.append(loss.data.item())
                        # if (epoch+1)%250 == 0:
                            # print('epoch: {}, train loss: {}, test loss: {}'.format(epoch+1, np.mean(train_loss), np.mean(test_loss)))
                    
                    # test
                    # model.eval()
                    MAPE_part = 0
                    update_count = 0
                    with torch.no_grad():
                        test_loss = []
                        for data in testloader_l:
                            nodepair,metric = data
                            nodepair = nodepair.float()
                            metric = metric.float()
                            nodepair = Variable(nodepair)
                            metric = Variable(metric)
                            if torch.cuda.is_available():
                                nodepair = nodepair.cuda()
                                metric = metric.cuda()
                            
                            out = model(nodepair)
                            loss = criterion(out, metric.reshape(metric.shape[0],1))
                            test_loss.append(loss.data.item())
                            
                            for update_id in range(len(out)):
                                if dataaug_label[j-2][update_count] == MAX:
                                    dataaug_label[j-2][update_count] = out[update_id]
                                    MAPE_part += abs((dataaug_label[j-2][update_count]-metric[update_id])/metric[update_id])
                                    update_count += 1
                                else:
                                    dataaug_label[j-2][update_count] = beta*dataaug_label[j-2][update_count]+(1-beta)*out[update_id]
                                    MAPE_part += abs((dataaug_label[j-2][update_count]-metric[update_id])/metric[update_id])
                                    update_count += 1
                    
                    MAPE = MAPE_part.cpu()*100/len(testloader_l)
                    LA_print[i][j] = MAPE
                    
                MAPE_part = sum(abs((dataaug_label[j-2]-monitor_l[3])/monitor_l[3]))
                MAPE = MAPE_part*100/dataaug_label[j-2].shape[0]
                LA_print[i][j] = MAPE
                LA_print[i][j+4] = MAPERL[j-2]
                
            # output
            for kk in range(10):
                if LA_print[i][kk] < LR_min[i][kk]:
                    LR_min[i][kk] = LA_print[i][kk]
                    LR[i][kk] = learning_rate
            
            print('------------------------------------------------')
            print('|AS{} MAPE(%) Type: {} Sampling Rate: {}|'.format(
                AS_print[AS_idx],LA[0],sampling_rate_print[i]))
            print('------------------------------------------------')
            print('|  Scheme  | RlMape |  Now   |  Min   |   LR   |')
            print('------------------------------------------------')
            for jj in range(6):
                if jj == 0 or jj == 1:
                    print('| {} |        | {:6.2f} | {:6.2f} | {:6.4f} |'.format(
                        sampling_type_print[jj],LA_print[i][jj],LR_min[i][jj],LR[i][jj]))
                elif jj == 5:
                    print('| {} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.4f} |'.format(
                        sampling_type_print[jj],MAPERL[jj-2],LA_print[i][jj],LR_min[i][jj],LR[i][jj]))
                    print('------------------------------------------------')
                else:
                    print('| {} | {:6.2f} | {:6.2f} | {:6.2f} | {:6.4f} |'.format(
                        sampling_type_print[jj],MAPERL[jj-2],LA_print[i][jj],LR_min[i][jj],LR[i][jj]))
