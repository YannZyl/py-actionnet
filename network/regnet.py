# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from .layers import conv2d, fc, softmax
from utils.datatools import get_next_batch, data_split

class RegNet(object):
    def __init__(self, cell_size=256, image_size=28, step_size=24, num_classes=3, lr=0.001,\
                        batch_size=16, test_batch=16, display_iters=100, test_iters=500, max_iterations=20000):
        self.cell_size = cell_size
        self.image_size = image_size
        self.step_size = step_size
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.test_batch = test_batch
        self.display_iters = display_iters
        self.test_iters = test_iters
        self.max_iterations = max_iterations
        self.history_left = np.zeros((1,step_size,image_size,image_size))
        self.history_right = np.zeros((1,step_size,image_size,image_size))
        self.build_model()
    
    def build_model(self):
        # input
        self.XL = tf.placeholder(tf.float32,shape=[None,self.step_size,self.image_size,self.image_size])
        self.XR = tf.placeholder(tf.float32,shape=[None,self.step_size,self.image_size,self.image_size])
        self.Y = tf.placeholder(tf.float32,shape=[None,self.num_classes])
        # output,loss function and optimizer
        self.output = self.net(self.XL,self.XR,reuse=None)
        self.loss_func = -tf.reduce_mean(tf.reduce_sum(self.Y*tf.log(self.output),1))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_func)
        # validation
        self.correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.output,1),tf.argmax(self.Y,1)),dtype=tf.float32))
        # test param
        self.norm_output = tf.cast(tf.greater_equal(self.output,tf.ones_like(self.output)*0.6), dtype=tf.float32)
        self.result = tf.argmax(self.norm_output, 1)
    
    def subnet_cnn(self, X, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # data reshape. (batch, step, height, width) --> (batch*step, height, width)
            X = tf.reshape(X,shape=[-1,self.image_size,self.image_size,1],name='reshape')
            # cnn net. input--conv--flatten--fc--drop
            conv1 = conv2d(X,output_dims=20,k_h=3,k_w=3,s_h=1,s_w=1,padding='SAME',name='conv')
            flatten = tf.reshape(conv1,[-1,self.image_size*self.image_size*20],name='flatten') 
            fc1 = fc(flatten,output_dims=128,name='fc')
            return fc1
    
    def subnet_rnn(self, X, name, reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            # reshape data
            X = tf.reshape(X,shape=[-1,self.step_size,128*2],name='cnneye_reshape')
            X = tf.transpose(X, perm=[1,0,2],name='cnneye_transpose')
            X = tf.reshape(X,shape=[-1,128*2],name='cnneye_reshape2')
            X = tf.split(X,self.step_size,0)
            # rnn network
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=0.7)
            output, state = tf.contrib.rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
            # softmax
            fc1 = fc(output[-1],output_dims=self.num_classes,name='fc')
            out = softmax(fc1,name='prob')
            return out
        
    def net(self, X_left, X_right, reuse=None): 
        with tf.variable_scope('RegNet', reuse=reuse):
            # CNN extract spatial feature
            out_cnn_l = self.subnet_cnn(X_left, name='cnn_left', reuse=None)
            out_cnn_r = self.subnet_cnn(X_right, name='cnn_right', reuse=None)
            # concate left eye feature and right eye feature
            out_cnn = tf.concat([out_cnn_l,out_cnn_r], axis=1, name='concate')
            # LSTM extract temporal feature
            out_rnn = self.subnet_rnn(out_cnn, name='rnn_eye', reuse=None)
            return out_rnn
    
    def train(self, session, data):
        # data separate
        self.left = data_split(X=data[0], train_rate=0.8)
        self.right = data_split(X=data[1], train_rate=0.8)
        self.label = data_split(X=data[2], train_rate=0.8)
        print('[Step2 Data] Train samples:{0}, valid samples:{1}'.format(len(self.left[0]), len(self.left[1])))
        # train
        for iters in range(self.max_iterations):
            batch_x_left = get_next_batch(self.left[0], None, iters, self.batch_size)
            batch_x_right = get_next_batch(self.right[0], None, iters, self.batch_size)
            batch_y = get_next_batch(self.label[0], None, iters, self.batch_size)
            loss, _ = session.run([self.loss_func,self.optimizer],feed_dict={self.XL:batch_x_left,self.XR:batch_x_right,self.Y:batch_y})
            if iters % self.display_iters == 0:
                print('[Step2 Train] RegNet Iters:{0}, loss:{1}'.format(iters,loss))
            if iters % self.test_iters == 0:
                self.val_internal(session)
    
    def val_internal(self, session):
        max_batches = len(self.left[1]) // self.test_batch
        rights = 0.0
        for iters in range(max_batches):
            batch_x_left = get_next_batch(self.left[1], None, iters, self.test_batch)
            batch_x_right = get_next_batch(self.right[1], None, iters, self.test_batch)
            batch_y = get_next_batch(self.label[1], None, iters, self.test_batch)
            sums = session.run(self.correct,feed_dict={self.XL:batch_x_left,self.XR:batch_x_right,self.Y:batch_y})
            rights += sums
        print('[Step2 Valid] RegNet val accuracy:{0}'.format(rights/len(self.left[1])))
    
    def predict(self, session, data):
        pred = session.run(self.result, feed_dict={self.XL:data[0], self.XR:data[1]})
        return pred
        
