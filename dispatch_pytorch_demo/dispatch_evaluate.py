#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:08:33 2019

@author: daniel
"""

import numpy as np
import pandas as pd #读取csv文件的库
import matplotlib.pyplot as plt
import torch
#from torch.autograd import Variable
import torch.optim as optim

data_path = 'bike-sharing-dataset/hour.csv'
rides = pd.read_csv(data_path)
rides.head()

dummy_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix = each, drop_first = False)
    rides = pd.concat([rides, dummies], axis = 1)
    
fields_to_drop = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 'instant', 'dteday', 'atemp', 'workingday']
data = rides.drop(fields_to_drop, axis = 1)

quant_features = ['cnt', 'temp', 'hum', 'windspeed']

scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] =(data[each]-mean)/std

test_data = data[-21*24:]
train_data = data[:-21*24]
print("训练数据：", len(train_data), '测试数据：', len(test_data))

# 将数据划分为特征列，与目标列
target_fields = ['cnt', 'casual', 'registered']
train_features, train_targets = train_data.drop(target_fields, axis = 1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis = 1), test_data[target_fields]
input_size = 56
hidden_size = 10
output_size = 1
batch_size = 128


import pickle
F = open('model.pkl', 'rb')
parameters = pickle.load(F)
weights = parameters['weights']
biases = parameters['biases']
weights2 = parameters['weights2']

def neu(x):
    hidden = x.mm(weights) + biases.expand(x.size()[0], hidden_size)
    hidden = torch.sigmoid(hidden)
    output = hidden.mm(weights2)
    return output

def cost(x, y):
    error = torch.mean((x-y)**2)
    return error

def optimizer_step(learning_rate):
    weights.data.add_(-learning_rate * weights.grad.data)
    biases.data.add_(-learning_rate * biases.grad.data)
    weights2.data.add_(-learning_rate * weights2.grad.data)
    
def zero_grad():
    if weights.grad is not None and biases.grad is not None and weights2.grad is not None:
        weights.grad.data.zero_()
        biases.grad.data.zero_()
        weights2.grad.data.zero_()
        
# 用训练好的神经网络在测试集上进行预测
targets = test_targets['cnt']
targets = targets.values.reshape([len(targets), 1])
targets = targets.astype('float')

x = torch.tensor(test_features.values, dtype = torch.double, requires_grad = True)
y = torch.tensor(targets, dtype = torch.double, requires_grad = True)

# 用神经网络进行预测
predict = neu(x)
predict = predict.data.numpy()
print(predict)

fig, ax = plt.subplots(figsize = (10, 7))

# 提取 cnt的均值和方差， 通过predict * std + mean还原cnt值
mean, std = scaled_features['cnt']
ax.plot(predict * std + mean, label='Prediction', linestyle = '--')
ax.plot(targets * std + mean, label='Data', linestyle = '-')
ax.legend()
ax.set_xlabel('Date-time')
ax.set_ylabel('Counts')
# 对横坐标轴进行标注
dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.savefig('predictions_plot.jpg')