#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:19:46 2017

@author: lhy
"""

from .data import Data
import numpy as np

class CIFAR_100(Data):
    
    def data_deal(self,file):
        dicts = self.unpickle(file)
        data = dicts[b'data']/1
        label = dicts[b'fine_labels']
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
        meann = [129.3, 124.1, 112.4]
        stdd = [68.2, 65.4, 70.4]
        for i in range(data.shape[3]):
            data[:,:,:,i] = (data[:,:,:,i] - meann[i])/stdd[i]
        return data