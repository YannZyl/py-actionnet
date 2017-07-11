#Introduction#
robot control via eye action based on cnn+lstm
```
step1: camera frame -- face detection(Haar+Adaboost)
step2: face region(step1) -- eyes detection(left+right, Haar+Adaboost)
step3: eyes region(step2) -- eyes selection(lenet-5)
step4: eyes region(frame difference or optical flow or original pic) -- cnn+lstm classification
step5: predict -- action control(finite state machine)
```
 
#Dependences#
numpy, scipy
```bash
sudo apt-get install python-dev, python-numpy, python-scipy
```
skimage
```bash
sudo apt-get install python-skimage  or  pip install scikit-image
```
sklearn
```bash
sudo apt-get install python-sklearn  or  pip install scikit-learn
```
opencv
```bash
sudo apt-get install python-opencv
```
tensorflow(https://www.tensorflow.org/install/)
    
