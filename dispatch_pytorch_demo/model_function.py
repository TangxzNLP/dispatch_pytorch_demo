#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 22:25:19 2019

@author: daniel
"""

import pickle
import torch 
from torch.autograd import Variable


def model_save(path, parameters):
    F = open('path/model.pkl','wb')
    pickle.dump(parameters, F) 
    ### pickle.dump(D, F)
    F.close()

def model_load(path, modelname):
    str = path + modelname
    parameters = open(str, 'rb')
    return parameters

class model_build:
    def __init__(self, weights, biases, weights2, input_size, hidden_size):
        self.weights = weights
        self.biases = biases
        self.weights2 = weights2
        self.input_size = input_size
        self.hidden_size = hidden_size
#    weights = torch.randn([input_size, hidden_size], dtype = torch.double, requires_grad = True)
#    biases = torch.randn([hidden_size], dtype = torch.double, requires_grad = True)
#    weights2 = torch.randn([hidden_size, output_size], dtype = torch.double, requires_grad = True)
    def neu(self, x):
        hidden = x.mm(self.weights) + self.biases.expand(x.size()[0], self.hidden_size)
        hidden = torch.sigmoid(hidden)
        output = hidden.mm(self.weights2)
        return output

    def cost(x, y):
        error = torch.mean((x-y)**2)
        return error

    def optimizer_step(self, learning_rate):
        self.weights.data.add_(-learning_rate * self.weights.grad.data)
        self.biases.data.add_(-learning_rate * self.biases.grad.data)
        self.weights2.data.add_(-learning_rate * self.weights2.grad.data)
    
    def zero_grad(self):
        if self.weights.grad is not None and self.biases.grad is not None and self.weights2.grad is not None:
            self.weights.grad.data.zero_()
            self.biases.grad.data.zero_()
            self.weights2.grad.data.zero_()