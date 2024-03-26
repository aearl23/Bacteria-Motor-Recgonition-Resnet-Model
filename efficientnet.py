import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.efficientnet import EfficientNet_B5_Weights
import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.nn.functional import smooth_l1_loss as SmoothL1Loss

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # No normalization required for bounding box regression
])

# Custom dataset class for training
# load the training images, train labels

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.data = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image_name = self.data.iloc[idx, 0]
        image = Image.open(img_path).convert('RGB')
        
        # Load bounding box annotations in csv format 
        x_center = self.data.iloc[idx, 1]
        y_center = self.data.iloc[idx, 2]
        width = self.data.iloc[idx, 3]
        height = self.data.iloc[idx, 4]
        motor_visible = self.data.iloc[idx, 5]

        annotation = torch.tensor([x_center, y_center, width, height, motor_visible])

        if self.transform: 
            image = self.transform(image)
        
        return image, image_name, annotation

# Model architecture for bounding box regression
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        
        #last layer of input features from classifier 
        num_ftrs = self.model.classifier[-1].out_features
        # probability of motor presence
        self.classification_head = nn.Linear(num_ftrs, 1)
        # bounding box coordinates and dimensions
        self.regression_head = nn.Linear(num_ftrs, 4)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        motor_presence = self.classification_head(x)
        bbox_params = self.regression_head(x)
        return motor_presence, bbox_params

# global variables for saving progress
checkpoint_dir = os.getcwd()
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')

# Load saved checkpoints
def load_checkpoint(model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    return start_epoch

# Define the training function
def train_model(model, train_loader, criterion_classification, criterion_regression, optimizer, device, num_epochs=10):
    start_epoch = load_checkpoint(model, optimizer)
    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss_classification = 0.0
        running_loss_regression = 0.0
        num_samples = 0  # Track the number of samples used for training
        
        for images, _, annotations in train_loader:
            images = images.to(device)
            annotations = annotations.to(device)
            
            optimizer.zero_grad()

            motor_presence, bbox_params = model(images)

            target_classification = annotations[:, -1].unsqueeze(1)  # Extract motor visibility as the target
            loss_classification = criterion_classification(motor_presence, target_classification.float())
            
            target_regression = annotations[:, :-1]  # Extract bounding box parameters as the target
            loss_regression = criterion_regression(bbox_params, target_regression.float())

            loss = loss_classification + loss_regression

            loss.backward()
            optimizer.step()
            
            running_loss_classification += loss_classification.item()  # Update running classification loss
            running_loss_regression += loss_regression.item()  # Update running regression loss
            
            num_samples += images.size(0)  # Update the number of samples used for training
        
        epoch_loss_classification = running_loss_classification / num_samples if num_samples > 0 else 0  # Calculate epoch classification loss
        epoch_loss_regression = running_loss_regression / num_samples if num_samples > 0 else 0  # Calculate epoch regression loss
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Classification Loss: {epoch_loss_classification:.4f}, Regression Loss: {epoch_loss_regression:.4f}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    print('Training complete.')

# Main function calling all parts 
    
def main():
    # Dataset paths
    train_annotations_file = 'train_labels.csv'
    train_img_dir = 'train'
    
    # Initialize dataset and data loader
    train_dataset = CustomDataset(train_annotations_file, train_img_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    model = BoundingBoxModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_regression = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion_classification, criterion_regression, optimizer, device, num_epochs=10)

if __name__ == '__main__':
    main()
