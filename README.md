# Introduction

robot control via eye action based on cnn+lstm
```
* step1: camera frame -- face detection(Haar+Adaboost)
* step2: face region(step1) -- eyes detection(left+right, Haar+Adaboost)
* step3: eyes region(step2) -- eyes selection(lenet-5)
* step4: eyes region(frame difference or optical flow or original pic) -- cnn+lstm classification
* step5: predict -- action control(finite state machine)
```
 
# Dependences
* numpy, scipy
```bash
sudo apt-get install python-dev, python-numpy, python-scipy
```
* skimage
```bash
sudo apt-get install python-skimage  or  pip install scikit-image
```
* sklearn
```bash
sudo apt-get install python-sklearn  or  pip install scikit-learn
```
* opencv
```bash
sudo apt-get install python-opencv
```
* [tensorflow](https://www.tensorflow.org/install/)


# Demo/Usage

Before run demo, you should train network with your own data, at first prepare your data. In file __train.py__, modify the param
```bash
face_cascade_file='data/model/haarcascade_frontalface_alt.xml',
eyes_cascade_file='data/model/haarcascade_eye.xml'
left_video_dir = 'data/video/left',
right_video_dir = 'data/video/right',
recover_video_path = 'data/video/recover/IMG_1950.MP4'
```
And then run the script __train.py__ to train the model
```bash
python train.py
```
At last, please run demo
```bash
python demo.py
```
