# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.datatools import prepare_data
from framework.eye_detect import EyeDetect
from framework.action_detect import ActionDetect

def train():
    # module1. CNN for eye image second detection
    eye_detect = EyeDetect(face_cascade_file='data/model/haarcascade_frontalface_alt.xml',
                           eyes_cascade_file='data/model/haarcascade_eye.xml',
                           left_video_dir = 'data/video/left',
                           right_video_dir = 'data/video/right',
                           recover_video_path = 'data/video/recover/IMG_1950.MP4',
                           image_size=28,
                           image_max_length=320,
                           batch_size=16,
                           num_classes=2, 
                           test_batch=32,
                           lr=0.0001, 
                           display_iters=500, 
                           test_iters=1000, 
                           max_iterations=20000)
                           
    # module2. CNN+LSTM for eye action detection
    action_detect = ActionDetect(step_size=32, 
                                 image_size=28,
                                 cell_size=256,
                                 num_classes=3,
                                 lr=0.0005,
                                 batch_size=16, 
                                 test_batch=16, 
                                 display_iters=100, 
                                 test_iters=500, 
                                 max_iterations=20000)
    # saver and memory config 
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=config) as session:
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        # module1 train step
        data1 = prepare_data(pos_dir='data/eyepos',neg_dir='data/eyeneg',re_size=28)
        eye_detect.train(session, data1)
        # generate module2 input
        data2 = eye_detect.output_lstm_format(session,step_size=32)
        # module2 train step
        action_detect.train(session, data2)
        # save model
        saver.save(session, 'data/model/tf_session_model.ckpt')

if __name__ == '__main__':
    train()