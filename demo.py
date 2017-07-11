# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy import ndimage
from utils.datatools import get_session
from framework.eye_detect import EyeDetect
from framework.action_detect import ActionDetect

# module1. CNN for eye image second detection
eye_detect = EyeDetect(face_cascade_file='data/model/haarcascade_frontalface_alt.xml',
                       eyes_cascade_file='data/model/haarcascade_eye.xml',
                       left_video_dir = 'data/video/left',
                       right_video_dir = 'data/video/right',
                       recover_video_path = 'data/video/recover/IMG_1950.MP4',
                       image_size=28, image_max_length=320)
                           
# module2. CNN+LSTM for eye action detection
action_detect = ActionDetect(step_size=32,image_size=28,cell_size=256)
# load session from exist file
session = get_session(model_dir='data/model')
# detect
num2str = {0:'recover',1:'Right',2:'Left'}
                            
def analyze_video(video_path, rotate_angle=0, drop_size=16):
    # previous left and right eye image, initial to zero
    pre_left, pre_right = np.zeros((28,28)), np.zeros((28,28))
    left, right = np.zeros((28,28)), np.zeros((28,28))
    prev_state = 0
    try:
        # analyze from video
        count = 0
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            # read frame one by one
            ret, frame = cap.read()
            # check frame state, wether none
            if frame is not None:
                # update index and rotate image if necessary
                count = min(count+1, drop_size)
                frame = ndimage.rotate(frame, rotate_angle)
                # module1: face detect -- eye detect -- eye select
                _, eyes_roi = eye_detect.detect_from_image(session, frame, True)
                # check detect result, if not none and index greater than 1, compute difference between two frames
                # update previous left and right eye frame
                if eyes_roi is not None and count > 1:
                    left = (eyes_roi[0]-pre_left)/255.0
                    right = (eyes_roi[1]-pre_right)/255.0
                    pre_left = eyes_roi[0]
                    pre_right = eyes_roi[1]
                else:
                    left = np.zeros_like(pre_left)
                    right = np.zeros_like(pre_right)
                # module2: eye action analyze, skip the front buffer size to erase error
                if count == drop_size:
                    pred = action_detect.predict(session,[left, right])
                    print('Action:', num2str[pred])
                    prev_state = pred
            # exit
            if (cv2.waitKey(1) & 0xFF == ord('q')) or frame is None:
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except:
        print('open video filed')

def analyze_camera(rotate_angle=0, buffer_size=16):
    # previous left and right eye image, initial to zero
    pre_left, pre_right = np.zeros((28,28)), np.zeros((28,28))
    left, right = np.zeros((28,28)), np.zeros((28,28))
    prev_state = 0
    # analyze from camera
    try:
        cap = cv2.VideoCapture(0)
        count = 0
        while(True):
            # read frame one by one
            ret, frame = cap.read()
            # check frame state, wether none
            if frame is not None:
                # update index and rotate image if necessary
                count = min(count+1, buffer_size)
                frame = ndimage.rotate(frame, rotate_angle)
                # module1: face detect -- eye detect -- eye select
                _, eyes_roi = eye_detect.detect_from_image(session, frame, True)
                # check detect result, if not none and index greater than 1, compute difference between two frames
                # update previous left and right eye frame
                if eyes_roi is not None and count > 1:
                    left = (eyes_roi[0]-pre_left)/255.0
                    right = (eyes_roi[1]-pre_right)/255.0
                    pre_left = eyes_roi[0]
                    pre_right = eyes_roi[1]
                else:
                    left = np.zeros_like(pre_left)
                    right = np.zeros_like(pre_right)
                # module2: eye action analyze, skip the front buffer size to erase error
                if count > buffer_size:
                    pred = action_detect.predict(session,[left, right])
                    if prev_state != pred:
                        print('Action:', num2str[pred])
                        prev_state = pred
            # exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    except:
        print('open camera filed')

if __name__ == '__main__':  
    analyze_video('data/video/test/testvideo.MP4', rotate_angle=270, drop_size=20)
