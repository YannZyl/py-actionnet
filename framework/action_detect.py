# -*- coding: utf-8 -*-
import numpy as np
from network.regnet import RegNet

class ActionDetect(object):
    def __init__(self, image_size=28, cell_size=256, step_size=24, num_classes=3, lr=0.001,batch_size=16, 
                         test_batch=16, display_iters=100, test_iters=500, max_iterations=50000): 
        # train params
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
        # predict params
        self.history_left = np.zeros((1,step_size,image_size,image_size))
        self.history_right = np.zeros((1,step_size,image_size,image_size))
        # decision params
        self.current_state = 0
        self.state_hold_time = 0
        self.max_state_hold_time = 5
        self.build_model()
        
    def build_model(self):
        self.action_detector = RegNet(cell_size=self.cell_size, 
                                      image_size=self.image_size, 
                                      step_size=self.step_size, 
                                      num_classes=self.num_classes, 
                                      lr=self.lr,
                                      batch_size=self.batch_size, 
                                      test_batch=self.test_batch, 
                                      display_iters=self.display_iters, 
                                      test_iters=self.test_iters, 
                                      max_iterations=self.max_iterations)
    
    # function: compute difference between an action sequence
    # input params: 
    #     data: set of action sequence, size: [#samples, #step size, image_h, image_w]            
    def frame_diff(self, data):
        left, right, label = data
        left_diff, right_diff = [], []
        # compute difference for every sequence
        for i in range(len(left)):
            left_seq, right_seq = left[i], right[i]
            diff_l, diff_r = np.zeros_like(left_seq), np.zeros_like(right_seq)
            # compute diffence, ignore the first frame, fill with zeros
            for j in range(1,len(left_seq)):
                diff_l[j] = (left_seq[j]-left_seq[j-1])/255.0
                diff_r[j] = (right_seq[j]-right_seq[j-1])/255.0
            left_diff.append(diff_l)
            right_diff.append(diff_r)
        left_diff, right_diff = np.array(left_diff), np.array(right_diff)
        return left_diff, right_diff, label
    
    # function: update model previous n state with data
    # input params:
    #      data: left and right sequence, size:[left, right], left/right:[#step size, image_h, image_w]                
    def update_history(self, data):
        left_image, right_image = data
        # update input
        x_left = np.zeros_like(self.history_left)
        x_left[0,0:-1,:,:] = self.history_left[0,1:,:,:]
        x_left[0,-1,:,:] = left_image
        x_right = np.zeros_like(self.history_right)
        x_right[0,0:-1,:,:] = self.history_right[0,1:,:,:]
        x_right[0,-1,:,:] = right_image
        # update history sequence
        self.history_left = x_left.copy()
        self.history_right = x_right.copy()
    
    def decision(self, next_state):
        # if current execute state equals to next predict state, hold state and reset hold time
        # else
        #   if current execute state is 0(none), change state right now and reset hold time
        #   else if two state is different and state hold time less than max hold time, ignore state, update hold time
        #   else if two state is different and state hold time equals to max hold time, change state right now and reset hold time
        if self.current_state == next_state:
            self.state_hold_time = 0
        elif self.current_state == 0:
            self.current_state = next_state
            self.state_hold_time = 0
        elif self.state_hold_time == self.max_state_hold_time:
            self.current_state = next_state
            self.state_hold_time = 0
        else:
            self.state_hold_time += 1
        return self.current_state
    
    # train RegNet
    def train(self, session, data):
        #data = self.frame_diff(data)
        self.action_detector.train(session, data)
    
    # predict function
    def predict(self, session, data):
        # update history sequence
        self.update_history(data)
        # predict
        test_data = [self.history_left, self.history_right]
        pred = self.action_detector.predict(session, test_data)
        # decide base on last n test
        pred = self.decision(pred[0])
        return pred
        