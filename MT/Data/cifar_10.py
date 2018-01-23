#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:05:45 2017
"""

from .data import Data
import numpy as np

class CIFAR_10(Data):
    
    def data_deal(self,file):
        dicts = self.unpickle(file)
        data = dicts[b'data']/1
        label = dicts[b'labels']
        data = np.reshape(data,(-1,3,32,32))
        data = np.transpose(data, (0, 2, 3, 1))
        label = np.reshape(label,(-1,))
        return data,label
    
    def load_data(self,adress,file):
        num = np.size(file)
        data,label = self.data_deal(adress+file[0])
        if num > 1: #train_batch
            for i in range(1,num):
                data2,label2 = self.data_deal(adress+file[i])
                data = np.append(data, data2, axis=0)
                label = np.append(label, label2, axis=0)
        return data,label
    
    def normal_by_constant(self,data):
        meann = [125.3, 123.0, 113.9]
        stdd = [63.0, 62.1, 66.7]
        for i in range(data.shape[3]):
            data[:,:,:,i] = (data[:,:,:,i] - meann[i])/stdd[i]
        return data    
