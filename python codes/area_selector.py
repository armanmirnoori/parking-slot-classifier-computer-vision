import cv2
import json

VIDEO_IN = "./data/parking_1920_1080.mp4"
OUT_JSON = "slots.json"

cap = cv2.VideoCapture(VIDEO_IN)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read first frame of video.")

slots = []  # list of (x, y, w, h)
drawing = False
x0, y0 = -1, -1
preview = frame.copy()

def redraw():
    global preview
    preview = frame.copy()
    for (x, y, w, h) in slots:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 1) #draw green line
    cv2.putText(preview, f"Slots: {len(slots)} | drag=add  u=undo  c=clear  s=save  q=quit",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def mouse(event, x, y, flags, param):
    global drawing, x0, y0, preview

    if event == cv2.EVENT_LBUTTONDOWN:  #starting point
        drawing = True
        x0, y0 = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw()
        x1, y1 = x, y
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 255), 1) #draw yellow line

    elif event == cv2.EVENT_LBUTTONUP: #ending point
        drawing = False
        x1, y1 = x, y
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        w = x_max - x_min
        h = y_max - y_min

        # ignore tiny accidental drags
        if w > 5 and h > 5:
            slots.append((x_min, y_min, w, h))
        redraw()

redraw()
cv2.namedWindow("Label Slots", cv2.WINDOW_NORMAL) #window normal allows to resize the window
cv2.resizeWindow("Label Slots", 1280, 720)  #set it to preview better 
cv2.setMouseCallback("Label Slots", mouse) #for mouse action, call mouse function

while True:
    cv2.imshow("Label Slots", preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("u") and slots:
        slots.pop()
        redraw()
    elif key == ord("c"):
        slots.clear()
        redraw()
    elif key == ord("s"):
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(slots, f)
        print(f"Saved {len(slots)} slots to {OUT_JSON}")
    elif key == ord("q"):
        break

cv2.destroyAllWindows()

