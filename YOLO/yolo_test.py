import hashlib
import cv2
from ultralytics import YOLO
import math

# model
model = YOLO("./trained_models/exp_yolo.pt")
classNames = ["PMF 1", "PMN 2", "F1", "RGD 5", "RKG 3", "TM 62", "OZM 72", "MON 50"]


def get_color_from_class_name(class_name):
    # Generate a hash for the class name
    hash_object = hashlib.md5(class_name.encode())
    hash_code = hash_object.hexdigest()
    
    # Convert the hash to an integer and map to BGR color
    color = tuple(int(hash_code[i:i+2], 16) for i in (0, 2, 4))
    return color


# Define input image
image_file = "pmn2_52.jpg"
img = cv2.imread(image_file)

results = model(img, stream=False)
# coordinates
for r in results:
    boxes = r.boxes
    for box in boxes:
        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        color = get_color_from_class_name(classNames[int(box.cls[0])])
        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        print("Confidence --->",confidence)

        # class name
        cls = int(box.cls[0])
        print("Class name -->", classNames[cls])

        if confidence > 0.5:
            # put box in cam
            color = (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5

            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)


cv2.imshow("Image", img)
cv2.waitKey(0)