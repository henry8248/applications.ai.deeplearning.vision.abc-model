# ABC model (Ancestral Background Classification)
## Description:
* The project uses two models to realize basic ABC, and the inference is done by building OpenCV with DL Streamer support.
* **This is the early version of my intern project. The latest version is confidential.**
>
## OS: Ubuntu 22.04LTS
>
## Datasets:
1. Fairface:https://github.com/joojs/fairface
2. UTKface: https://susanqq.github.io/UTKFace/
3. Vggface2: https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b
> 
## Models:
1. Model1: YuNet(For face-detection) Related link: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
2. Model2: The ABC model which can recognize four categories. That is, A: African, C: Caucasian, S: South Asian, and E: East Asian. Model's val-accuracy: about 83%
>
## Model features:
1. YuNet's input format: RGB frames
2. ABC model input format: Grayscale 336x336 NV12 frames
## Inference:
1. Set up a python virtual env in terminal 
2. Build openCV with DL Streamer support: https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c
3. Connect your USB camera to the computer and check its supported image format by v4l2-ctl command
   (Refer to https://www.mankier.com/1/v4l2-ctl)
4. Clone the repository
5. Modify the path of external or internal cameras for gstreamer pipeline in Race-recognition/gstreamer/Inference.py to connect the camera to the pipeline
6. Execute!
>
## Examples:

![image](https://github.com/henry8248/The_ABC_model/blob/main/demo/Biden.png)
![image](https://github.com/henry8248/The_ABC_model/blob/main/demo/Obama.png)

