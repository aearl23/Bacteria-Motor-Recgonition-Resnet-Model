import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(pretrained=True)

num_classes = 2  # Assuming you have 2 classes: background and flagella
model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes, pretrained=True)

images = '' #extract all image filepaths from .txt file
for image in images:
    output = model(image)
