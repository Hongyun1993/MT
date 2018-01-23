#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:34:42 2017

"""

from .data import Data
import numpy as np
import scipy.io as sio

class SVHN(Data):
    
    def load_data(self,adress,file):
        train_data = sio.loadmat(adress+file)
        data = train_data['X']/1
        data = np.transpose(data, (3, 0, 1, 2))
        label = train_data['y']
        index = (label == 10)
        label[index] = 0
        return data,label
