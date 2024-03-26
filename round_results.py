import pandas as pd

# Load the CSV file
df = pd.read_csv('submission.csv')

# Round the bounding box parameters to whole numbers and convert them to integers
df[['x_Center', 'y_Center', 'width', 'height']] = df[['x_Center', 'y_Center', 'width', 'height']].round().astype(int)

# Save the rounded results to a new CSV file
df.to_csv('rounded_submission.csv', index=False)