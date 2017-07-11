# -*- coding: utf-8 -*-
import cv2
from glob import iglob
from scipy import ndimage
from utils.nms import nms

face_detector = cv2.CascadeClassifier('model/haarcascade_frontalface_alt.xml')   
eyes_detector = cv2.CascadeClassifier('data/model/haarcascade_eye.xml')
eye_size = 28

def resize(image, image_max_length=320):
    width, height = image.shape[1], image.shape[0]
    scale = image_max_length / max(width, height)
    image = cv2.resize(image,(int(width*scale),int(height*scale)))
    return image, scale
    
def detect(full_image, visualize=False):
    # step1: preprocess, image resize, grayscale, equalizeHist
    image, _ = resize(full_image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    # face detect
    face_regions = face_detector.detectMultiScale(image_gray, 1.05, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (60,60))
    # eyes detect based on face region
    eyes_regions = []
    for (x, y, w, h) in face_regions:
        face_image = image_gray[y:y+h,x:x+w]
        # resize face, transform to high resolution
        face_image, scale = resize(face_image)
        # first eyes detection
        eyes_roi = eyes_detector.detectMultiScale(face_image, 1.05, 2, cv2.cv.CV_HAAR_SCALE_IMAGE, (60,60),(120,120))
        # second eyes detection  
        for (ex, ey, ew, eh) in eyes_roi:
            eyes_regions.append([x+int(ex/scale),y+int(ey/scale),int(ew/scale),int(eh/scale)]) 
    # apply nms reduce bbox        
    eyes_regions = nms(eyes_regions, thres=0.5)
    # visulize
    if visualize:
        # plot
        for (x, y, w, h) in face_regions:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for (x, y, w, h) in eyes_regions:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)   
        # 显示标定框
        cv2.imshow("eye detect", image)
        cv2.waitKey(5) 
    eyes_region = [cv2.resize(image_gray[y:y+h,x:x+w],(eye_size, eye_size)) for (x,y,w,h) in eyes_regions]
    return eyes_region

# sampling method1: sample from single image
def sample_from_image(image_path, rotate_angle=0, output_dir=None):
    image = cv2.imread(image_path, 0)
    image = ndimage.rotate(image, rotate_angle)
    eyes_roi = detect(image, False)
    if output_dir is not None:
        for idx, eye in enumerate(eyes_roi):
            cv2.imwrite(output_dir+'/eye_{0}.jpg'.format(idx), eye)
    return eyes_roi
    
# sampling method2: sample from image folder
def sample_from_image_folder(image_folder,rotate_angle=0, output_dir=None):
    eye_images = []
    for image_path in iglob(image_folder+'/*'):
        eye = sample_from_image(image_path, rotate_angle, output_dir)
        eye_images += eye
    return eye_images
    
# sampling method3: sample from single video    
def sample_from_video(video_path, rotate_angle=0, output_dir=None):
    count = 0
    eyes = []
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            count += 1
            frame = ndimage.rotate(frame, rotate_angle)
            eyes_roi = detect(frame, visualize=True)
            eyes += eyes_roi
        if (cv2.waitKey(1) & 0xFF == ord('q')) or frame is None:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    if output_dir is not None:
        for idx, eye in enumerate(eyes_roi):
            cv2.imwrite(output_dir+'/eye_{0}.jpg'.format(idx), eye)
    return eyes_roi
    
# sampling method4: samples from video folder
def sample_from_video_folder(video_folder,rotate_angle=0, output_dir=None):
    eye_images = []
    for video_path in iglob(video_folder+'/*'):
        eye = sample_from_video(video_path, rotate_angle, output_dir)
        eye_images += eye
    return eye_images
    
if __name__ == '__main__':
    sample_from_video_folder('data/video/left',rotate_angle=270, output_dir='output')