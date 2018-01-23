#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:08:12 2017

"""

import numpy as np
import random

class Data:
    def __init__(self,normal_method, use_mt,augment, batchsize, classes, adress, file):
        self.data,self.label = self.load_data(adress,file)
        self.data_size = self.data.shape[0] 
        self.mt_data_size = int(self.data_size/2)
        self.classes = classes
        self.datas = self.normalize(self.data,normal_method)
        self.labels = self.one_hot(self.label)
        self.new_data = self.datas
        self.new_label = self.labels
        self.batch_counter = 0
        self.batchsize = batchsize
        self.normal_method = normal_method
        self.augment = augment
        self.use_mt = use_mt
        self.adress = adress
    
    def one_hot(self,label):
        labels = np.zeros([self.data_size, self.classes])
        for i in range(self.data_size):
            labels[i][label[i]] = 1
        return labels
        
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
        
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
    
    def normalize(self,data,normal_method):
        if normal_method =='divide_256':
            data = data/256
        elif normal_method =='by_channel':
            data = self.normal_by_channel(data)
        elif normal_method =='constant':
            data = self.normal_by_constant(data)
        return data
    
    def normal_by_channel(self,data):
        for i in range(data.shape[3]):
            meann = np.mean(data[:,:,:,i])
            stdd = np.std(data[:,:,:,i])
            data[:,:,:,i] = (data[:,:,:,i] - meann)/stdd
        return data
    
    def normal_by_constant(self,data):
        return data
    
    
    def suffling(self,data,label):
        index = np.random.permutation(self.data_size)
        data = data[index]
        label = label[index]
        return data,label
    
    def next_epoch(self):
        data,label = self.suffling(self.datas,self.labels)
        if self.augment == True:
            data = self.augmentation(data,4)
        if self.use_mt == True:
            data,label = self.MT(data,label)
        self.new_data = data
        self.new_label = label
        self.batch_counter = 0
    
    def next_batch(self):
        start = self.batch_counter*self.batchsize
        end = (self.batch_counter+1)*self.batchsize
        next_batch_data = self.new_data[start:end]
        next_batch_label = self.new_label[start:end]
        self.batch_counter += 1
#        if next_batch_data.shape[0] < self.batchsize:
#            self.next_epoch()
#            self.next_batch()
        return next_batch_data, next_batch_label
        
    
    def MT(self,data,label):
        half = int(self.mt_data_size)
        data1 = data[:half]
        label1 = label[:half]
        data2 = data[half:]
        label2 = label[half:]
        new_data = np.zeros(data1.shape)
        new_label = np.zeros(label1.shape)
        randd = np.random.rand(half)
        for i in range(half):
            new_data[i] = randd[i]*data1[i]+(1-randd[i])*data2[i]
            new_label[i] = randd[i]*label1[i]+(1-randd[i])*label2[i]
        return new_data,new_label
        
    def augment_image(self,image,pad):
        init_shape = image.shape
        new_shape = [init_shape[0] + pad*2, init_shape[1] + pad*2, init_shape[2]]
        zero_pad = np.zeros(new_shape)
        zero_pad[pad:init_shape[0]+pad, pad:init_shape[1]+pad, :] = image
        init_x = np.random.randint(0,pad*2)
        init_y = np.random.randint(0,pad*2)
        cropped = zero_pad[init_x:init_x + init_shape[0],init_y:init_y + init_shape[1],:]
        flip = random.getrandbits(1)
        if flip:
            cropped = cropped[:, ::-1,:]
        return cropped

    def augmentation(self,initial_images,pad):
        new_images = np.zeros(initial_images.shape)
        data_size = self.data_size            
        for i in range(data_size):
            new_images[i] = self.augment_image(initial_images[i],pad = pad)
        return new_images            
            
        
        
        
