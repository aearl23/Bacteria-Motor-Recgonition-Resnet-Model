import cv2
import os
import pandas as pd

# Function to draw bounding box
def draw_bbox(image, x_center, y_center, width, height, color=(0, 255, 0), thickness=2):
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# Read the data from the file
data = pd.read_csv('train_labels.csv')

# Create output directory if it doesn't exist
labeled_images = 'output_images'
os.makedirs(labeled_images, exist_ok=True)

# Iterate through each row
for index, row in data.iterrows():
    image_name = row['external_id']
    x_center = row['x_center']
    y_center = row['y_center']
    width = row['width']
    height = row['height']
    motor_visible = row['motor_visible']

    # Skip images where motor is not visible
    if motor_visible == 0:
        continue

    # Load the image
    image_path = os.path.join('train', image_name)
    image = cv2.imread(image_path)

    # Draw bounding box if center and dimensions are valid
    if x_center != -1 and y_center != -1 and width != -1 and height != -1:
        draw_bbox(image, x_center, y_center, width, height)

    # Save the image with bounding box
    output_path = os.path.join(labeled_images, os.path.basename(image_name))
    cv2.imwrite(output_path, image)

print("Images with bounding boxes saved to:", labeled_images)