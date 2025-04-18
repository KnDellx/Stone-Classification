### How to Generate submission.csv from test_loader


#### 1. **Define the Prediction Function**  
Use the following function to extract predictions from `test_loader`:
```python
def predict(model, loader, device):
model.eval()  # Set the model to evaluation mode
predictions = []  # Store predicted classes
image_ids = []    # Store image filenames

with torch.no_grad():  # Disable gradient computation
   for images, img_paths in tqdm(loader, desc="Predicting on test set"):
       images = images.to(device)  # Move images to the specified device
       outputs = model(images)     # Forward pass to get model outputs
       _, predicted = torch.max(outputs, 1)  # Get predicted classes
       
       # Collect predictions and image IDs
       predictions.extend(predicted.cpu().numpy())
       image_ids.extend([os.path.basename(path) for path in img_paths])

return image_ids, predictions
```

#### 2. **Run Predictions**  
Call the prediction function with the trained model, `test_loader`, and device:
```python
image_ids, predictions = predict(model, test_loader, device)
```

#### 3. **Create the Submission File**  
```python
import pandas as pd
import os

# Create DataFrame
submission_df = pd.DataFrame({
   "id": image_ids,    # Image filenames
   "label": predictions  # Predicted classes
})

# Save to the specified path
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Kaggle submission file saved to {submission_path}")
```

### Output Description
- **`submission.csv` Format**:  
The file contains two columns:
- `id`: Filenames of test images (without paths, e.g., `image1.jpg`).
- `label`: Predicted class indices (e.g., 0, 1, 2, depending on the number of classes).


- **Example Content**:
```
id,label
000001.jpg,0
000002.jpg,1
000003.jpg,2
```
