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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # No normalization required for bounding box regression
])

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
# Custom dataset class for training
# load the training images, train labels

class CustomValidationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform: 
            image = self.transform(image)

        return image, img_name
    
def main():
    validation_set = CustomValidationDataset(
        'validation',
        transform = transform,
    )

    validation_load = DataLoader(validation_set, batch_size = 32, shuffle=False)

    model = BoundingBoxModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_path = 'model_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    results = []

    with torch.no_grad():
        for images, img_names in validation_load:
            images = images.to(device)
            outputs = model(images)

            motor_presence, bbox_params = outputs

             # Flatten the output tensor and convert to numpy array
            predictions = motor_presence.cpu().numpy().flatten()
            bbox_predictions = bbox_params.cpu().numpy()
            
            # Append image filenames and predictions
            for img_name, prediction, bbox_prediction in zip(img_names, predictions, bbox_predictions):
                if prediction >= 0.5:
                    results.append([img_name] + bbox_prediction.tolist() + [1])
                else:
                    results.append([img_name] + [-1,-1,-1,-1,0])

    # Define column headers for the CSV file
    headers = ['external_id', 'x_center', 'y_center', 'width', 'height', 'motor_visible']
    
    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=headers)

    df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    main()