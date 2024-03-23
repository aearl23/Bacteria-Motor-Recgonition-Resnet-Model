import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np

# Define transformations for data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # No normalization required for bounding box regression
])

# Custom dataset class for training
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_paths = [line.strip() for line in open(annotations_file)]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load bounding box annotations and normalize them
        # Assuming the format is [x_center, y_center, width, height]
        annotation = np.array([0.5, 0.5, 0.5, 0.5])  # Placeholder for demonstration
        return image, annotation

# Define the model architecture for bounding box regression
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super(BoundingBoxModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)
        self.fc = nn.Linear(self.model._fc.in_features, 4)  # Output: [x_center, y_center, width, height]

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# Define the training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, annotations in train_loader:
            images = images.to(device)
            annotations = annotations.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, annotations)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    print('Training complete.')

# Main function
def main():
    # Dataset paths
    train_annotations_file = 'train_annotations.txt'
    train_img_dir = 'train_images'
    
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
    train_model(model, train_loader, criterion, optimizer)

if __name__ == '__main__':
    main()