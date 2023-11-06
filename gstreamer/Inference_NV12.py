
# """
# How to optimize: cv2.dnn() 
# 1. Raise framerate[solved]        #Model pruning/ Enable gpu support
# 2. Raise model accuracies: Indonesian -> Black/White, face-detection's cannot be adopted to
# smaller faces and tilted faces
# 3. Add more logs for the race model -> Latino? South-Asian?
# 4. Add UI by tkinter [TBD]
# 5. Plug webcam and make it work [solved]
# 6. Enforce UI-> pyQT, Django 
# 7. Launch to different platforms and input source formats -> Android/Chromebook, Line, Discord
# """
### if the race model can be fed with grayscale data[solved], the model might be inserted into original gvaclassify pipeline
### with face-detection adas used. 
import cv2 #opencv for gstreamer
import numpy as np


#gst_str = "videotestsrc ! videoconvert ! appsink drop=1"
#Asian_speech.mp4, face-detection.mp4 test.mp4 white_mexican.mp4 w_b_a.mp4 black_indian_white.mp4

pb  = "//home//venus3//anaconda3//envs//race_training//frozen_models_nv12//frozen_graph83%.pb"
model_path="/home/venus3/anaconda3/envs/streamer/models_videos/yunet.onnx"
              
face_detector = cv2.FaceDetectorYN_create(model_path, "", (0, 0))

racenet = cv2.dnn.readNetFromTensorflow(model=pb)   


# gst_str = "filesrc location=/home/venus3/anaconda3/envs/streamer/models_videos/test.mp4 ! decodebin ! video/x-raw, width=640, height=360 ! videoconvert ! video/x-raw, format=BGR ! appsink"

# gst_str = "gst-launch-1.0 -v v4l2src device=/dev/video1 io-mode=2 ! video/x-raw, format=NV12, framerate=30/1, width=1280, height=720 ! videoconvert ! video/x-raw, format=BGR ! appsink"
gst_str = '''gst-launch-1.0 -v v4l2src device=/dev/video1 io-mode=2 ! 
video/x-raw, format=NV12, width=1280, height=720 ! videoflip method=4 ! appsink'''
# gst_str = "filesrc location=/home/venus3/anaconda3/envs/streamer/models_videos/black_indian_white.mp4 ! decodebin ! video/x-raw, format=BGR, width=1280, height=720 ! appsink"
# videoflip method=4
#NV12(YUV420)

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
race = {1: "C", 0: "A", 3: "S", 2: "E"}
# race = {0 : "A", 1: "C", 2: "E", 3: "S"}
factor = 5
fontScale = 0.7
timer = 0
def converter(img):
    B, G, R = np.squeeze(np.split(img, 3, -1))
            
    rows, cols = R.shape

    y = 0.1826*R + 0.6142*G + 0.0620*B + 16
    u = -0.1006*R - 0.3386*G + 0.4392*B + 128
    v = 0.4392*R - 0.3989*G - 0.0403*B + 128

    shrunk_u = (u[0::2, 0::2] + u[1::2, 0::2] + u[0::2, 1::2] + u[1::2, 1::2])*0.25 #select all u, calculate the u's avg
    shrunk_v = (v[0::2, 0::2] + v[1::2, 0::2] + v[0::2, 1::2] + v[0::2, 1::2])*0.25
    uv = np.zeros((rows//2, cols))

    uv[:, 0::2] = shrunk_u
    uv[:, 1::2] = shrunk_v

    nv12 = np.vstack((y, uv))
    nv12 = np.round(nv12).astype("uint8")

    return nv12

while cap.isOpened():
   
    
    try:
        ret, frame = cap.read()
        
        
        # print(frame2.shape)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        # frame2 = frame2[0:224, :]
        face_detector.setInputSize((frame2.shape[1], frame2.shape[0]))
        # face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        
        _, faces = face_detector.detect(frame2)
        faces = faces if faces is not None else []
        timer += 1
        
        for face in faces:
            box = face[:4].astype(np.int32)
            cv2.rectangle(frame2, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), thickness=2, color=(255, 255, 0))
            img = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            
            # if timer == 3:
            #     timer = 0
            try:

                    # nv12 = cv2.resize(img, (336, 336))
                    # frame = frame.reshape((336, 336, 1))
                # img = converter(img)
                    # cv2.imshow("Faces", nv12)
                if timer in list(range(50, 70)):
                    
                    img3 = cv2.dnn.blobFromImage(img, size=(336, 336), crop=True)
                    racenet.setInput(img3)
                    cvOut = racenet.forward()
                    key = np.squeeze(cvOut)
                    key_sorted = np.sort(key)
                    index = np.argsort(-key)
                    # ratio = round(max(key)*100)
                    
                    cv2.putText(frame2, f"{race[index[0]]}: {round(key_sorted[-1]*100)}%; {race[index[1]]}: {round(key_sorted[-2]*100)}%", (box[0]-factor, box[1]-factor), cv2.FONT_HERSHEY_SIMPLEX, 
                                fontScale, color=(0, 255, 0), thickness=2)
                elif timer > 70:
                    timer = 0
            except:
                
                continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(frame, f"fps: {fps}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale, color=255, thickness=1) 
        cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)
        cv2.imshow("Faces", frame2)

        if not ret:
            print("fail")
            break

    except KeyboardInterrupt:
        break
    
    if cv2.waitKey(1) == ord('q'):
       break
cap.release()
cv2.destroyAllWindows()


