#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:57:38 2017

@author: lhy
"""

import tensorflow as tf

class Densenet:
    def __init__(self, input_height = 32, input_width = 32, input_depth = 3,output_dimension = 10, K = 12, n1 = 12, n2 = 12, n3 = 12):
         self.input_height = input_height
         self.input_width = input_width
         self.input_depth = input_depth
         self.output_dimension = output_dimension
         self.K = K
         self.n1 = n1
         self.n2 = n2
         self.n3 = n3
         self.feature_num = 16
         weight_decay = 1e-4
         channels = 3  # input channels
         
         self.keep_prob = tf.placeholder(tf.float32,shape=[])
         self.lr = tf.placeholder(tf.float32,shape=[])
         self.istrain = tf.placeholder(tf.bool,shape=[])
         
         with tf.name_scope('inputs'):
            self.inputs = tf.placeholder(tf.float32,[None,self.input_height,self.input_width,channels],name = 'inputs') #[data_num,height,width]
            self.labels = tf.placeholder(tf.float32,[None,self.output_dimension],name = 'labels')
         self.kernel = tf.Variable(tf.truncated_normal([3,3,channels,self.feature_num],stddev=0.01))  
         self.feature_map = tf.nn.conv2d(self.inputs, self.kernel, strides = [1,1,1,1], padding = 'SAME')

         with tf.name_scope('block1'):
             self.block_out,self.feature_num = block(self.feature_map,self.n1,self.K,self.feature_num,self.keep_prob,self.istrain)
             self.transition_out,self.feature_num = transition(self.block_out,self.K,self.feature_num,self.keep_prob,self.istrain)

         with tf.name_scope('block2'):
             self.block_out,self.feature_num = block(self.transition_out,self.n2,self.K,self.feature_num,self.keep_prob,self.istrain)
             self.transition_out,self.feature_num = transition(self.block_out,self.K,self.feature_num,self.keep_prob,self.istrain)

         with tf.name_scope('block3'):
             self.block_out,self.feature_num = block(self.transition_out,self.n1,self.K,self.feature_num,self.keep_prob,self.istrain)
         self.block_out = tf.contrib.layers.batch_norm(self.block_out, scale=True, is_training=self.istrain, updates_collections=None)
         self.block_out = tf.nn.relu(self.block_out)
         self.block_out = tf.nn.avg_pool(self.block_out, [ 1,8, 8, 1 ], [1, 8, 8, 1 ], 'VALID')
         self.block_out = tf.reshape(self.block_out, [-1, self.feature_num])
         with tf.name_scope('complete_connect'):            
             self.W = tf.Variable(tf.truncated_normal([self.feature_num,output_dimension],stddev=0.01),name = 'outputW') #size maybe is not correct
             self.bias = tf.Variable(tf.truncated_normal([output_dimension],stddev=0.01),name = 'outputbias') 
             self.outputs = tf.matmul(self.block_out,self.W)+self.bias

         with tf.name_scope('loss'):
             cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.outputs,labels = self.labels))
             l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
             self.loss = cross_entropy +l2*weight_decay

         with tf.name_scope('train'):
             self.train_step = tf.train.MomentumOptimizer(self.lr,0.9,use_nesterov=True).minimize(self.loss)
    

def block(input,n,k,feature_num,keep_prob,istrain):
    feature_map = input    
    for i in range(n):
        layername = 'layer%s'%i
        with tf.name_scope(str(i)):
            feature_map,feature_num = dense_connect(feature_map,k,feature_num,layername,keep_prob,istrain)
    return feature_map,feature_num
    
def dense_connect(input,k,feature_num,layername,keep_prob,istrain):
    feature_map = tf.contrib.layers.batch_norm(input, scale=True, is_training=istrain, updates_collections=None)
    feature_map = tf.nn.relu(feature_map)
    W = tf.Variable(tf.truncated_normal([3,3,feature_num,k],stddev=0.01),name = 'W')
    output = tf.nn.conv2d(feature_map, W, strides = [1,1,1,1], padding = 'SAME')
    output = tf.nn.dropout(output,keep_prob)   
    output = tf.concat([input, output],3)  # dense connect
    feature_num += k
    return output,feature_num

    
def transition(input,k,feature_num,keep_prob,istrain):
    feature_map = tf.contrib.layers.batch_norm(input, scale=True, is_training=istrain, updates_collections=None)
    output = tf.nn.relu(feature_map)
    W = tf.Variable(tf.truncated_normal([1,1,feature_num,feature_num],stddev=0.01), name = 'W')
    output = tf.nn.conv2d(output, W,strides = [1,1,1,1], padding = 'SAME')
    output = tf.nn.dropout(output,keep_prob)    
    output = tf.nn.avg_pool(output, [ 1, 2, 2, 1 ], [1, 2, 2, 1], padding = 'VALID')
    return output,feature_num

