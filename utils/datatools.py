# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
from glob import glob

__all__ = ['get_session',
           'prepare_data',
           'data_split',
           'get_next_batch',
           'data_arguement',
           'video_cut_argue']

def get_session(model_dir):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    session = tf.Session(config=config)    
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver = tf.train.Saver()  
    if ckpt is not None:
        print('Checkpoint exists, load from file')
        saver.restore(session,ckpt.model_checkpoint_path)
        tf.get_variable_scope().reuse_variables()
    else:
        print('Checkpoint is not exists, create and initialize variables')
        session.run(tf.global_variables_initializer())
    return session

def prepare_data(pos_dir=None, neg_dir=None, re_size=28):
    X_pos, Y_pos, X_neg, Y_neg = [], [], [], []
    for path in glob(pos_dir+'/*.jpg'):
        X_pos.append(cv2.resize(cv2.imread(path,0),(re_size,re_size))) 
        Y_pos.append([0,1])
    for path in glob(neg_dir+'/*.jpg'):
        X_neg.append(cv2.resize(cv2.imread(path,0),(re_size,re_size))) 
        Y_neg.append([1,0])
    X = X_pos + X_neg
    Y = Y_pos + Y_neg
    # shuffle data
    index = np.random.permutation(len(X))
    X = np.array([X[i] for i in index]).reshape(-1,re_size,re_size,1)
    Y = np.array([Y[i] for i in index])
    return X, Y

def data_split(X, Y=None, train_rate=0.8):
    train_nums = int(len(X) * train_rate)
    X_train = X[0:train_nums]
    X_val = X[train_nums:]
    if Y is not None:
        Y_train = Y[0:train_nums]
        Y_val = Y[train_nums:]
        return X_train, Y_train, X_val, Y_val
    else:
        return X_train, X_val
        
def get_next_batch(X, Y=None, offset=0, batch_size=16):
    max_batches = len(X) // batch_size
    offset = offset % max_batches
    batch_x = X[offset*batch_size : (offset+1)*batch_size]
    if Y is not None:
        batch_y = Y[offset*batch_size : (offset+1)*batch_size]    
        return batch_x, batch_y
    else:
        return batch_x
    
def data_arguement(left_right_sequence_sets, seq_max_length=24):
    new_left, new_right = [],[]
    for [sequence_l, sequence_r] in left_right_sequence_sets:
        frame_length = len(sequence_l)
        if frame_length > seq_max_length:
            # if current video frames greater than max frame length, then truncation!
            dis = frame_length - seq_max_length
            # truncation method1: right truncation. [0-seq_max_length] selected, [seq_max_length-frame_length] drop
            left_truncate_l, left_truncate_r = sequence_l[dis:dis+seq_max_length], sequence_r[dis:dis+seq_max_length]
            # truncation method2: mid truncation. [0-dis/2] drop, [dis/2-dis/2+seq_max_length] selected, [dis/2+seq_max_length-frame_length] drop
            mid_truncate_l, mid_truncate_r = sequence_l[dis//2:dis//2+seq_max_length],sequence_r[dis//2:dis//2+seq_max_length]
            # truncation method3: left truncation. [0-dis] drop, [dis-dis+seq_max_length] selected
            right_truncate_l,right_truncate_r = sequence_l[0:seq_max_length], sequence_r[0:seq_max_length]
            # appending
            new_left.append(left_truncate_l)
            new_left.append(mid_truncate_l)
            new_left.append(right_truncate_l)
            new_right.append(left_truncate_r)
            new_right.append(mid_truncate_r)
            new_right.append(right_truncate_r)
        else:
            # else fill the video sequence with zero frames
            dis = seq_max_length - frame_length
            # padding method1: left padding.
            left_padding_l, left_padding_r = [np.zeros_like(sequence_l[0])]*seq_max_length, [np.zeros_like(sequence_r[0])]*seq_max_length
            left_padding_l[dis:], left_padding_r[dis:] = sequence_l, sequence_r
            # padding method2: middle padding
            mid_padding_l,  mid_padding_r = [np.zeros_like(sequence_l[0])]*seq_max_length, [np.zeros_like(sequence_r[0])]*seq_max_length
            mid_padding_l[dis//2:dis//2+frame_length],mid_padding_r[dis//2:dis//2+frame_length] = sequence_l, sequence_r
            # padding method3: right padding
            right_padding_l,  right_padding_r = [np.zeros_like(sequence_l[0])]*seq_max_length, [np.zeros_like(sequence_r[0])]*seq_max_length
            right_padding_l[0:frame_length], right_padding_r[0:frame_length]= sequence_l, sequence_r
            # appending
            new_left.append(left_padding_l)
            new_left.append(mid_padding_l)
            new_left.append(right_padding_l)
            new_right.append(left_padding_r)
            new_right.append(mid_padding_r)
            new_right.append(right_padding_r) 
    print('Data arguement. before:{0},after:{1}'.format(len(left_right_sequence_sets), len(new_left)))
    return new_left, new_right

def video_cut_argue(left_right_sequence, cut_length=18, cut_stride=4, argue=True, seq_max_length=24):
    new_left, new_right = [], []
    left_sequence, right_sequence = left_right_sequence
    # compute cut times
    max_time = (len(left_sequence)-cut_length)//cut_stride+1
    for i in range(max_time):
        cut_left_sequence = left_sequence[cut_stride*i : cut_stride*i+cut_length]
        cut_right_sequence = right_sequence[cut_stride*i : cut_stride*i+cut_length]
        new_left.append(cut_left_sequence)
        new_right.append(cut_right_sequence)
    # if data is small, data arguement
    if argue:
        left_right_sequence_sets = [[new_left[i],new_right[i]] for i in range(len(new_left))]
        new_left, new_right = data_arguement(left_right_sequence_sets, seq_max_length=seq_max_length)
    return new_left, new_right