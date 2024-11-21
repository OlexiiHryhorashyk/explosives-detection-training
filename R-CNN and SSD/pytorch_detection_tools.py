from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

class_names = ["PMF 1", "PMN 2", "F1", "RGD 5", "RKG 3", "TM 62", "OZM 72", "MON 50"]

def get_predictions(model, image_path, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)

    # Filter predictions based on the threshold
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    print(pred_boxes)
    print(pred_labels)
    print(pred_scores)
    
    filtered_boxes = pred_boxes[pred_scores > threshold]
    filtered_labels = pred_labels[pred_scores > threshold]
    filtered_scores = pred_scores[pred_scores > threshold]
    
    return filtered_boxes, filtered_labels, filtered_scores

def visualize_predictions(image_path, boxes, labels, scores):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box, label, score in zip(boxes, labels, scores):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 1)
        text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(image, text, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
