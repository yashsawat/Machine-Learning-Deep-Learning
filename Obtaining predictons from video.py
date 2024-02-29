import os

from ultralytics import YOLO
import cv2

VIDEOS_DIR = os.path.join('.', 'video')
video_path = os.path.join(VIDEOS_DIR, 'testing.mp4')
video_path_out = '{}_out1.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.','grapes/sugarcane/banana.pt')
model = YOLO(model_path)  # Load a custom model

threshold = 0.2

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Define colors for different classes
            if results.names[int(class_id)] == 'weed':
                color = (0, 0, 255)  # Red color for weed
            elif results.names[int(class_id)] == 'grapes/sugarcane/banana':
                color = (0, 255, 0)  # Green color for sugarcane

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
