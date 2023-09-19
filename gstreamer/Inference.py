
# """
# How to optimize: cv2.dnn() 
# 1. Raise framerate        #Model pruning/ Enable gpu support
# 2. Raise model accuracies: Indonesian -> Black/White, face-detection's cannot be adopted to
# smaller faces and tilted faces
# 3. Add more logs for the race model -> Latino? South-Asian?
# 4. Add UI by tkinter [TBD]
# 5. Plug webcam and make it work [TBD]
# 6. Enforce UI-> pyQT, Django
# 7. Launch to different platforms and input source formats -> Android/Chromebook, Line, Discord
# """
### if the race model can be fed with grayscale data, the model might be inserted into original gvaclassify pipeline
### with face-detection adas used. 
import cv2 #opencv for gstreamer
import numpy as np

#haarcascade_profileface.xml 
# clf = cv2.CascadeClassifier() 
#gst_str = "videotestsrc ! videoconvert ! appsink drop=1"
#Asian_speech.mp4, face-detection.mp4 test.mp4 white_mexican.mp4 w_b_a.mp4 black_indian_white.mp4

pb  = "//home//venus3//anaconda3//envs//race_training//frozen_models//race_frozen_graph.pb"
# pb = "/home/venus3/anaconda3/envs/race_training/84%_race_model/saved_model.pb"
model_path="/home/venus3/anaconda3/envs/streamer/models&videos/yunet.onnx"
                    
face_detector = cv2.FaceDetectorYN_create(model_path, "", (0, 0))

racenet = cv2.dnn.readNetFromTensorflow(model=pb)   


# gst_str = "filesrc location=/home/venus3/anaconda3/envs/streamer/models&videos/test.mp4 ! decodebin ! video/x-raw ! queue ! videoconvert ! video/x-raw, format=BGR ! appsink"
# image/jpg format=YUY2
gst_str = "gst-launch-1.0 -v v4l2src device=/dev/video2 io-mode=2 ! video/x-raw, format=NV12, framerate=30/1, width=1280, height=720 ! videoconvert ! video/x-raw, format=BGR ! appsink"
# gst_str = "gst-launch-1.0 -v v4l2src device=/dev/video1 ! video/x-raw, format=YUV2, framerate=30/1, image/jpeg, width=848,height=480 ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
race = {0 : "Asian", 1: "Black", 2: "Indian", 3: "White"}
factor = 5
fontScale = 1

while cap.isOpened():
    try:
        ret, frame = cap.read()
        
        
        frame = cv2.flip(frame, 1)
        face_detector.setInputSize((frame.shape[1], frame.shape[0]))

        _, faces = face_detector.detect(frame)
        faces = faces if faces is not None else []
        for face in faces:
            box = face[:4].astype(np.int32)
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), thickness=2, color=(255, 255, 0))
           
            img = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
            
            try:
                racenet.setInput(cv2.dnn.blobFromImage(np.float32(img/255), size=(224, 224), swapRB=True, crop=False))
                cvOut = racenet.forward()
                cv2.putText(frame, race[np.squeeze(cvOut).argmax()], (box[0]-factor, box[1]-factor), cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale, color=(0, 255, 0), thickness=2)
            except:
                continue
        
        cv2.namedWindow("Faces", cv2.WINDOW_NORMAL)
        cv2.imshow("Faces", frame)
        # cv2.imshow("test", frame)
        if not ret:
            print("fail")
            break
    except KeyboardInterrupt:
        break
    
    if cv2.waitKey(1) == ord('q'):
        
        break
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"fps: {fps}")
cap.release()
cv2.destroyAllWindows()