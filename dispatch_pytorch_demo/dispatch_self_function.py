#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:37:37 2019

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

"""
    (一). 数据预处理，包括：a,类型变量 one-hot化, 剔除无用特征数据以及多余特征数据；b, 数值变量标准化； c.对数据进行train, test划分
"""

"""
a 类型变量: 有 season, mnth, hr, weekday, weathersit(天气情况),这5个变量的数值的递增并不对应信号强度的增大，所以要做one-hot化处理，
这里会用到 panda中的pd.get_dummies(), 如dummy = pd.get_dummies(rides['season'], prefix = season, drop_first = false)
使得season一列变成season1, season2, season3, season4, 再通过pd.concat([rides, dummy], axis = 1) 将one-hot化的内容加入到
rides中，但是season 会被保留下来，再通过 rides.drop('season', axis = 1)将该列剔除. 
season ranges from 1~4
mnth   ranges from 1~12
hr     ranges from 1~24
weekday ranges from 0~6
weathersit ranges from 1~4
"""
#即将要one-hot处理的列，label名
dummy_fields = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix = each, drop_first = False)
    rides = pd.concat([rides, dummies], axis = 1)
    
# 将原有类型变量对应的列去掉，同时把一些不相关的列（特征）去掉
fields_to_drop = ['season', 'mnth', 'hr', 'weekday', 'weathersit', 'instant', 'dteday', 'atemp', 'workingday']
data = rides.drop(fields_to_drop, axis = 1)
# 查看现在信息
data.head()

"""
b 数值变量: 有 cnt, temp, hum, windspeed. 因为每个变量都是相互独立的，所以他们绝对值大小与问题本身没有关系，为了消除数值之间的差异，我们
对每一个数值变量进行标准化处理。使其数值在0左右波动。比如，temp，它在整个数据库中取值mean(temp), 方差为std(temp).
"""
quant_features = ['cnt', 'temp', 'hum', 'windspeed']
# 将每个变量的均值和方差都存储到scaled_features变量中
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] =(data[each]-mean)/std

"""
c 对数据集进行分割， 将所有的数据集分为测试集和训练集。 将以后21天数据一共 21*24个数据点作为测试集，其它是训练集
"""
test_data = data[-21*24:]
train_data = data[:-21*24]
print("训练数据：", len(train_data), '测试数据：', len(test_data))

# 将数据划分为特征列，与目标列
target_fields = ['cnt', 'casual', 'registered']
train_features, train_targets = train_data.drop(target_fields, axis = 1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis = 1), test_data[target_fields]

# 将数据转化为numpy格式
X = train_features.values
Y = train_targets['cnt'].values
Y = Y.astype(float)

Y = np.reshape(Y, [len(Y),1])
losses = []
train_features.head()


"""
    (二). 构建神经网络并进行训练， 分别两种构建方法： a, 手动编写用Tensor运算的神经网络； b, 调用Pytorch现有函数，构建序列化神经网络
"""

"""
a, 手写用Tensor运算的神经网路
    已知的数据：X.shape[]=(16875, 56), Y.shape = (16875, 1)
    构建网络的参数：
                input_size = X.shape[1],也就是一组特征向量的纬度，这里是56.
                hidden_size = 10, 设置10隐含节点
                output_size = 1, 1个输出节点
                batch_size = 128, 设置批处理大小为 128组向量
                则(三层神经网络：输入，隐含，输出)：
                    weights.shape = (56, 10)
                    biases.shape = (,10)
                    weights2.shape = (10, 1)
"""
input_size = X.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128

weights = torch.randn([input_size, hidden_size], dtype = torch.double, requires_grad = True)
biases = torch.randn([hidden_size], dtype = torch.double, requires_grad = True)
weights2 = torch.randn([hidden_size, output_size], dtype = torch.double, requires_grad = True)


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

# 神经网络的训练

losses = []



for i in range(1000):
    batch_loss = []
    # start 和 end 分别是提取一个batch的起始和终止下标
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        # 将 X 的每一个batch tensor化
        xx = torch.tensor(X[start:end], dtype = torch.double, requires_grad = True)
        yy = torch.tensor(Y[start:end], dtype = torch.double, requires_grad = True)
        predict = neu(xx)
        loss = cost(predict, yy) 
        loss.backward()
        optimizer_step(0.01)       
        zero_grad()
        batch_loss.append(loss.data.numpy())
    # 每隔100步输出一下损失值
    if i % 100 ==0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))
        batch_loss =[]

# 打印输出损失值
fig = plt.figure(figsize = (10, 7))
plt.plot(np.arange(len(losses)) * 100, losses, 'o--')
plt.xlabel('epoch')
plt.ylabel('MSE')

import pickle
F = open('model.pkl','wb')
parameters = {}
parameters['weights'] = weights
parameters['biases'] = biases
parameters['weights2'] = weights2
pickle.dump(parameters, F) 
### pickle.dump(D, F)
F.close()

"""
测试神经网络
"""
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