from coco_dataset_reader import CocoDetectionWithTransform
from torchvision import transforms
import torch
from torchvision.models.detection import ssd, _utils, ssdlite
import torch.nn.functional as F  # Alias for functional utilities

def collate_fn(batch):
    return tuple(zip(*batch))

def get_custom_dataset(train_data_dir: str = "") -> CocoDetectionWithTransform:
    # Paths to your dataset and annotations
    if len(train_data_dir) == 0:
        train_data_dir = "./vehicles_dataset/No_Apply_Grayscale/No_Apply_Grayscale/Vehicles_Detection.v8i.coco/train"
    train_coco = f"{train_data_dir}/_annotations.coco.json"
    # Data transformations
    transform = transforms.Compose([transforms.ToTensor()])
    # Custom dataset instance
    my_dataset = CocoDetectionWithTransform(
        root=train_data_dir,
        annFile=train_coco,
        transform=transform
    )
    return my_dataset

def save_model(model, path, optimizer, epoch, loss):
    model_state_path = path
    torch.save(model.state_dict(), model_state_path)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, model_state_path)
    
def create_ssd_model(num_classes=8, size=640):
    # Load the Torchvision pretrained model.
    model = ssd.ssd300_vgg16()
    # Retrieve the list of input channels. 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Ensure the model is on the correct device

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Compute the loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            
            # Get the predictions
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            print("Predictions:", pred.view(-1).tolist())
            print("Targets:", target.tolist())
            
            # Count correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average loss over the entire test dataset
    test_loss /= len(test_loader.dataset)

    # Print accuracy and loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
