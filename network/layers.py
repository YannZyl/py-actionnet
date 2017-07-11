# -*- coding: utf-8 -*-
import tensorflow as tf

__all__ = ['conv2d', 
           'relu', 
           'lrn', 
           'max_pool', 
           'fc', 
           'dropout', 
           'softmax_with_loss', 
           'softmax']


def conv2d(input_tensor, output_dims, k_h=3, k_w=3, s_h=2, s_w=2, padding='VALID', name='conv2d'):
    with tf.variable_scope(name):
        fan_in = input_tensor.get_shape().as_list()[-1] * k_h * k_w
        limit = 3.0 * 2.0 / fan_in
        weights = tf.get_variable('weights', shape=[k_h,k_w,input_tensor.shape[-1],output_dims], dtype=tf.float32,\
                                    initializer=tf.random_uniform_initializer(-limit,limit))
        bias = tf.get_variable('bias', shape=[output_dims], dtype=tf.float32,\
                                    initializer=tf.random_uniform_initializer(-limit,limit))
        out = tf.nn.conv2d(input_tensor, weights, [1,s_h,s_w,1], padding)
        out = tf.nn.bias_add(out, bias)
        return out

def relu(input_tensor, name='relu'):
    return tf.nn.relu(input_tensor, name)

def lrn(input_tensor, local_size=5, alpha=0.0005, beta=0.75, k=2, name='lrn'):
    return tf.nn.local_response_normalization(input_tensor, depth_radius=local_size, \
                                        bias=k, alpha=alpha, beta=beta, name=name)

def max_pool(input_tensor, k_h, k_w, s_h, s_w, padding='VALID', name='pool'):
    return tf.nn.max_pool(input_tensor, ksize=[1,k_h,k_w,1], strides=[1,s_h,s_w,1],\
                            padding=padding, name=name)

def fc(input_tensor, output_dims, name='fc'):
    with tf.variable_scope(name):
        fan_in = input_tensor.get_shape().as_list()[-1]
        limit = 3.0*2.0/fan_in
        weights = tf.get_variable('weights', shape=[input_tensor.shape[-1],output_dims], dtype=tf.float32,\
                                    initializer=tf.random_uniform_initializer(-limit,limit))
        bias = tf.get_variable('bias', shape=[output_dims], dtype=tf.float32, \
                                initializer=tf.random_uniform_initializer(-limit,limit))
        out = tf.matmul(input_tensor,weights)
        out = tf.add(out, bias)
        return out

def dropout(input_tensor, keep_prob=0.5, name='drop'):
    return tf.nn.dropout(input_tensor, keep_prob=keep_prob, name=name)

def softmax(input_tensor, name='softmax'):
    return tf.nn.softmax(input_tensor, name=name)
    
def softmax_with_loss(input_tensor, label, name='loss'):
    return tf.nn.softmax_cross_entropy_with_logits(input_tensor, label, name=name)