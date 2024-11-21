import torch

from torch.utils.data import DataLoader
import torchvision

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from training_tools import collate_fn, get_custom_dataset, save_model

model_state_path = '../trained_models/exp_faster_rcnn.pth'

print("Loading dataset...")
train_data_dir = "../explosives_dataset/coco/train"
my_dataset = get_custom_dataset(train_data_dir)
print('Number of samples: ', len(my_dataset))
# DataLoader instance
data_loader = DataLoader(my_dataset,
                         batch_size=2,
                         shuffle=True,
                         collate_fn=collate_fn)

# Initialize the model
print("Loading the model...")
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
num_classes = 9
# Update the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# load last fitted model state
checkpoint = torch.load(model_state_path)
model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)


# for images, targets in data_loader:
#     losses = model(images, targets)
#     loss = sum(loss for loss in losses.values())
#     print("Poop")
# save_model(model, model_state_path, optimizer, 0, loss)


optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

epoch = checkpoint['epoch']
last_loss = checkpoint['loss']
min_loss = last_loss

num_epochs = 1
requested_epochs = epoch+num_epochs
print(f"Started training model for {num_epochs}: {epoch}->{requested_epochs} epochs total.")
# Training loop
while epoch <= requested_epochs:
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        i += 1
    epoch+=1
    # save the model if loss is smaller
    if loss < min_loss:
        save_model(model, model_state_path, optimizer, epoch, loss)
        min_loss=loss
    lr_scheduler.step()


print("Training complete!")
