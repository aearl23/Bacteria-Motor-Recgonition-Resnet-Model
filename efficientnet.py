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
import torchvision

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

        annotation = [x_center, y_center, width, height, motor_visible]

        if self.transform: 
            image = self.transform(image)
        
        return image, image_name, annotation

# Model architecture for bounding box regression
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.DEFAULT)
        
        #last layer of input features from classifier 
        num_ftrs = self.model.classifier[-1].in_features
        
        self.output_head = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        x = self.model(x)
        x = self.output_head(x)
        return x

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
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    start_epoch = load_checkpoint(model, optimizer)
    model.to(device)
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, annotations in train_loader:
            images = images.to(device)
            annotations = annotations.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs[:, 1:], annotations[:, :5])
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
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

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)

if __name__ == '__main__':
    main()