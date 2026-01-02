import cv2
import json
import pickle 
from skimage.transform import resize
import numpy as np

model_path = "model.p"
slots_path = "slots.json"
video_in = "./data/parking_1920_1080.mp4"
video_out = "output_video.mp4"

model = pickle.load(open(model_path, "rb"))

with open (slots_path, "r", encoding="utf-8") as f:
    slots = json.load(f)

labels = {0 : "empty", 1: "not_empty"}

def preprocess_patch(patch_bgr):
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
     #crop the image and convert bgr to rgb as a float number and devide with 255 to get 0<num<1


    patch_small = resize(patch_rgb, (32,32), anti_aliasing = True)
    #resize the image like svm model we trained, anti aliasing make it smoother

    return patch_small.flatten().reshape(1,-1)
    #make it 2D in 1 row


#capture the video 
cap = cv2.VideoCapture(video_in)
if not cap.isOpened():
    raise RuntimeError(f"cannot open video: {video_in}")

# prepare the output video format

fps = cap.get(cv2.CAP_PROP_FPS) or 30 #get the fps of video or if not, defualt = 30

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #video width which is 1920
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #video height which is 1080

fourcc = cv2.VideoWriter_fourcc(*"mp4v") #prepare mp4 format for saving
out = cv2.VideoWriter(video_out, fourcc, fps, (W,H))


#4) run frame by frame

while True:
    ret , frame = cap.read()
    if not ret:
        break

    empty_count = 0
    full_count = 0

    for (x,y,w,h) in slots: #loop through each parking slot rectangle
        x2,y2 = x+w, y+h  #calculate bottom-right corner of the rectangle
        if x < 0 or y<0 or x2 > W or y2 > H: #check the condition if rectangle is insade the frame
            continue

        patch = frame[y:y2, x:x2]
        pred = model.predict(preprocess_patch(patch))[0]

        #convert prediction to label 
        if isinstance(pred, (int, np.integer)):
            label = labels.get(int(pred), str(pred))
        else:
            label = str(pred)

        #counting 
        if label == "empty":
            empty_count+= 1
            color = (0,0,255)
        else:
            full_count+= 1
            color = (0, 255, 0)

        #draw rectangle and label the video
        cv2.rectangle(frame, (x,y), (x2,y2), color, 2)

    summary = f"empty: {empty_count} full: {full_count} total : {len(slots)}"
    cv2.putText(frame, summary, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


    #show live resualt
    cv2.imshow("parking slot", frame)

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ("q"):
        break

cap.release() 
out.release()
cv2.destroyAllWindows()