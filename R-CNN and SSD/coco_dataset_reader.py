import torch 
import torchvision.datasets as datasets
# Custom dataset class to transform COCO annotations

class CocoDetectionWithTransform(datasets.CocoDetection):
    def __getitem__(self, index):
        img, target = super(CocoDetectionWithTransform, self).__getitem__(index)
        
        # Apply transform to the image
        if self.transform is not None and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        
        # Processing the target (annotations)
        boxes = []
        labels = []
        
        for obj in target:
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Ensure there are no empty boxes
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        
        return img, target