#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:00:17 2017

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

from Data.cifar_10 import CIFAR_10
from Data.cifar_100 import CIFAR_100
from Data.svhn import SVHN

from Model.densenet import Densenet


#__Main__
def main():
    use_MT = True
    normal_method = 'constant' #'divide_256','by_channel','constant'
    batchsize = 64
    maxmax = 301
    #______________________________train_densenet_by_CIFAR-10_______________________________________
    data_type = 'CIFAR-10'
    train_network(data_type,use_MT,normal_method,batchsize,maxmax)
    #______________________________train_densenet_by_CIFAR-10+______________________________________
#    data_type = 'CIFAR-10+'
#    train_network(data_type,use_MT,normal_method,batchsize,maxmax)
#    #______________________________train_densenet_by_CIFAR-100______________________________________
#    data_type = 'CIFAR-100'
#    normal_method = 'by_channel'
#    train_network(data_type,use_MT,normal_method,batchsize,maxmax)
#    #______________________________train_densenet_by_CIFAR-100+_____________________________________
#    data_type = 'CIFAR-100+'
#    normal_method = 'constant' 
#    train_network(data_type,use_MT,normal_method,batchsize,maxmax)
#    #______________________________train_densenet_by_SVHN___________________________________________
#    data_type = 'SVHN'
#    normal_method = 'divide_256'
#    maxmax = 41
#    train_network(data_type,use_MT,normal_method,batchsize,maxmax)
  
   
def train_network(data_type,use_MT,normal_method,batchsize,maxmax):  
    print('-'*10+data_type+'-'*10)
    print('load_data...')     
    #data_load-------------------------------------------------------------------------------------- 
    if data_type == 'CIFAR-10':
        keep_prob = 0.8
        adress = '/home/lhy/下载/cifar-10-batches-py/'
        train_file = [ 'data_batch_%d' % d for d in range(1, 6) ] 
        test_file = [ 'test_batch' ]    
        classes = 10   
        #train_data
        augment = None
        use_mt = use_MT
        data_train = CIFAR_10(normal_method, use_mt, augment, batchsize, classes, adress, train_file)
        #test_data
        augment = None
        use_mt = None
        batchsize = 100
        data_test = CIFAR_10(normal_method, use_mt, augment, batchsize, classes, adress, test_file) 
        
    if data_type == 'CIFAR-10+':
        keep_prob = 1
        adress = '/home/lhy/下载/cifar-10-batches-py/'
        train_file = [ 'data_batch_%d' % d for d in range(1, 6) ] 
        test_file = [ 'test_batch' ]      
        classes = 10    
        #train_data
        augment = True
        use_mt = use_MT
        data_train = CIFAR_10(normal_method, use_mt, augment, batchsize, classes, adress, train_file)
        #test_data
        augment = None
        use_mt = None
        batchsize = 100
        data_test = CIFAR_10(normal_method, use_mt, augment, batchsize, classes, adress, test_file) 
    
    if data_type == 'CIFAR-100':
        keep_prob = 0.8
        adress = '/home/lhy/下载/cifar-100-python/'
        train_file = ['train']
        test_file = ['test']
        classes = 100  
        #train_data
        augment = None
        use_mt = use_MT
        data_train = CIFAR_100(normal_method, use_mt, augment, batchsize, classes, adress, train_file)
        #test_data
        augment = None
        use_mt = None
        batchsize = 100
        data_test = CIFAR_100(normal_method, use_mt, augment, batchsize, classes, adress, test_file)

    if data_type == 'CIFAR-100+':
        keep_prob = 1
        adress = '/home/lhy/下载/cifar-100-python/'
        train_file = ['train']
        test_file = ['test']
        classes = 100 
        #train_data
        augment = True
        use_mt = use_MT
        data_train = CIFAR_100(normal_method, use_mt, augment, batchsize, classes, adress, train_file)
        #test_data
        augment = None
        use_mt = None
        batchsize = 100
        data_test = CIFAR_100(normal_method, use_mt, augment, batchsize, classes, adress, test_file) 
    
    if data_type == 'SVHN':
        keep_prob = 0.8
        adress = '/home/lhy/下载/SVHN/'
        train_file = 'train_32x32.mat'
        test_file = 'test_32x32.mat'    
        classes = 10 
        #train_data
        augment = None
        use_mt = use_MT
        data_train = SVHN(normal_method, use_mt, augment, batchsize, classes, adress, train_file)
        #test_data
        augment = None
        use_mt = None
        batchsize = 100
        data_test = SVHN(normal_method, use_mt, augment, batchsize, classes, adress, test_file)                   
    #train_param----------------------------------------------------------------------------
    learning_rate = 0.2
    train_trace = np.zeros([maxmax,2]) #[loss,accuracy]
    test_trace = np.zeros([maxmax,2]) 
    time_trace = np.zeros(maxmax)      
    #model_param----------------------------------------------------------------------------    
    input_height = 32
    input_width = 32
    input_depth = 3 
    output_dimension = classes
    K = 12 
    n1 = 12 
    n2 = 12 
    n3 = 12                
    model = Densenet(input_height,input_width,input_depth,output_dimension,K,n1,n2,n3)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)       
    #iterate-------------------------------------------------------------------------------  
    print('begin_iteration...')
    total_start_time = time.time()
    for i in range(maxmax):
        start_time = time.time()
        if data_type == 'SVHN':
            if i == 20: learning_rate = 0.02
            if i == 30: learning_rate = 0.002 
        else:
            if i == 150: learning_rate = 0.02
            if i == 225: learning_rate = 0.002       
        #train-----------------------------------------------------------------------------
        los_train = 0
        acc_train = 0 
        if data_train.use_mt == True:
            num = int(data_train.mt_data_size/data_train.batchsize)
        else:
            num = int(data_train.data_size/data_train.batchsize)  
            
        data_train.next_epoch()
        for ii in range(num):  
            input_data, input_label = data_train.next_batch()    
            loss_train,accuracy_train = train_a_batch(sess, model, input_data, input_label, learning_rate, keep_prob)
            los_train += loss_train
            acc_train += accuracy_train
        loss_train = los_train/num
        accuracy_train = acc_train/num        
        #test------------------------------------------------------------------------------       
        los_test = 0
        acc_test = 0        
        num = int(data_test.data_size/data_test.batchsize)  
        data_test.next_epoch()
        for ii in range(num):
            input_data,input_label = data_test.next_batch()
            loss_test,accuracy_test = test_a_batch(sess, model, input_data, input_label)
            los_test += loss_test
            acc_test += accuracy_test        
        loss_test = los_test/num
        accuracy_test = acc_test/num        
        #record----------------------------------------------------------------------------        
        train_trace[i][0] = loss_train
        test_trace[i][0]  = loss_test
        train_trace[i][1] = accuracy_train
        test_trace[i][1]  = accuracy_test
        print('-'*20+str(i)+'-'*20)
        print ('train_all:',loss_train,accuracy_train)
        print ('test_all:',loss_test,accuracy_test)        
        draw_result(maxmax,train_trace, test_trace)
        time_per_epoch = time.time() - start_time
        time_left = int(maxmax - i)*time_per_epoch
        print('Time per epoch: %s, \n left time: %s'%(str(timedelta(seconds = time_per_epoch)),                                                   str(timedelta(seconds = time_left))))
        total_train_time = time.time() - total_start_time
        print('Total training time: %s'%str(timedelta(seconds = total_train_time)))
        time_trace[i] = total_train_time        
    #save-----------------------------------------------------------------------------------       
    save_result(data_type,maxmax,time_trace,train_trace, test_trace)  
    

def draw_result(maxmax,train_trace, test_trace):
    x = range(maxmax)
    #plot
    plt.plot(x,train_trace[:,1], 'r')
    plt.plot(x,train_trace[:,0], 'r--')
    plt.plot(x,test_trace[:,1], 'b',)
    plt.plot(x,test_trace[:,0], 'b--')
    plt.grid(color='b',linestyle='--')
    plt.show()
    plt.legend()
    plt.pause(0.01)
    plt.clf()

def save_result(name,maxmax, time_trace, train_trace, test_trace):
    np.savetxt( name + '_train.csv',train_trace,delimiter = ',')
    np.savetxt( name + '_test.csv',test_trace,delimiter = ',')  
    np.savetxt( name + '_time.csv',time_trace,delimiter = ',') 
    x = range(maxmax)
    plt.figure()    
    plt.plot(x,train_trace[:,1], 'r',label='accuracy_train')
    plt.plot(x,test_trace[:,1], 'b',label='accuracy_test')
    plt.grid(color='b',linestyle='--')
    plt.savefig( name + '.png')

def train_a_batch(sess,model,input_data, input_label, learning_rate, keep_prob):
    feed_dict_op = {model.inputs: input_data, model.labels:input_label, model.lr: learning_rate, model.keep_prob: keep_prob,model.istrain: True}
    _,outputs_train,loss_train = sess.run([model.train_step,model.outputs,model.loss],feed_dict = feed_dict_op)
    correct_prediction = np.equal(np.argmax(outputs_train, 1), np.argmax(input_label,1))
    accuracy_train = np.mean(correct_prediction.astype(float))  
    return loss_train, accuracy_train

def test_a_batch(sess,model,input_data, input_label):
    feed_dict_op = {model.inputs: input_data, model.labels: input_label, model.keep_prob: 1, model.istrain: False}
    outputs_test,loss_test = sess.run([model.outputs,model.loss],feed_dict = feed_dict_op)  
    correct_prediction = np.equal(np.argmax(outputs_test, 1), np.argmax(input_label, 1))
    accuracy_test = np.mean(correct_prediction.astype(float))
    return loss_test, accuracy_test    
    

main()

    
    

