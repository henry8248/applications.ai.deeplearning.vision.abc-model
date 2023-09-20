# Race-recognition
##Description:
* The project uses two models to realize basic race-recognition, and the inference is done by building OpenCV with DL Streamer support.
>
##OS: Ubuntu 22.04LTS
>
##Datasets:
1. Fairface:https://github.com/joojs/fairface
2. UTKface: https://susanqq.github.io/UTKFace/
3. Vggface2: https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b
> 
##Models:
1. Model1: YuNet(For face-detection) Related link: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
2. Model2: The Race model which can recognize four races. That is, Black, White, Indian, and Asian.
>
##Inference:
1. Set up a python virtual env in terminal 
2. Build openCV with DL Streamer support: https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c
3. Connect your USB camera to the computer and check its supported image format by v4l2-ctl command
   (Refer to https://www.mankier.com/1/v4l2-ctl)
4. Clone the repository
5. Modify the path of external or internal cameras for gstreamer pipeline in Race-recognition/gstreamer/Inference.py to connect the camera to the pipeline
6. Execute!
>
##Examples: (.mp4 files are also supported)
![image](https://github.com/henry8248/Race-recognition/blob/main/demo/success.png)
![image](https://github.com/henry8248/Race-recognition/blob/main/demo/USB_cam_demo.png)
    


