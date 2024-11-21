import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from training_tools import collate_fn, get_custom_dataset, save_model, create_ssd_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

model_state_path = '../trained_models/exp_ssd.pth'

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
num_classes = 9
model = create_ssd_model(num_classes)
# load last fitted model state
checkpoint = torch.load(model_state_path)
model.load_state_dict(checkpoint["model_state_dict"])


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# lr_scheduler = StepLR(optimizer, step_size=2)
optimizer = optim.Adam(params, lr=0.0001)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

# Load checkpoint details
epoch = checkpoint['epoch']
last_loss = checkpoint['loss']
min_loss = last_loss

num_epochs = 1
requested_epochs = epoch + num_epochs

print(f"Started training model for {num_epochs} epochs: {epoch} -> {requested_epochs}.")

# Training loop
while epoch <= requested_epochs:
    model.train()
    total_loss = 0  # Track total loss for the epoch

    for i, (images, targets) in enumerate(data_loader):
        # Move data to the device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients
        optimizer.zero_grad()

        # Compute losses and backpropagate
        losses = model(images, targets)
        loss = sum(loss for loss in losses.values())
        loss.backward()
        optimizer.step()

        # Accumulate loss for logging and scheduler
        total_loss += loss.item()

        # Log progress every 2 iterations
        if i % 2 == 0:
            print(f"Epoch {epoch}/{requested_epochs}, Iteration {i}, Loss: {loss.item():.4f}")

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

    # Update scheduler with the average loss
    lr_scheduler.step(avg_loss)

    # Save model checkpoint
    save_model(model, model_state_path, optimizer, epoch, avg_loss)

    # Update epoch
    epoch += 1

print("Training complete!")