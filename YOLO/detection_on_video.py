from ultralytics import YOLO
import cv2
import math 
import hashlib

# Open the video file
video_path = "./vehicles_dataset/SampleVideo_LowQuality.mp4"  # Change this to your video file path
cap = cv2.VideoCapture(video_path)

# Model
model = YOLO("my_models/yolov8n_fitted.pt")

# Object classes
classNames = ['', '', 'Car', 'Motorcycle', 'Pickup', 'Bus', '', 'Truck']

def get_color_from_class_name(class_name):
    # Generate a hash for the class name
    hash_object = hashlib.md5(class_name.encode())
    hash_code = hash_object.hexdigest()
    
    # Convert the hash to an integer and map to BGR color
    color = tuple(int(hash_code[i:i+2], 16) for i in (0, 2, 4))
    return color

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            cls = int(box.cls[0])
            print("Class number", cls)
            color = get_color_from_class_name(classNames[cls])
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # Class name
            print("Class name -->", classNames[cls])
            # print("Class name -->", class_name)

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
