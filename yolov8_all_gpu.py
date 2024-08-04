import os
import torch
import torch.quantization
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import scale_image
from torch.utils.data import Dataset, DataLoader

# Load your dataset
data = check_det_dataset('/media/parashuram/dataset/YOLOv8_/datasets//data.yaml')

# Create a subset for calibration
calibration_dataset = data['train'][:1000]  # Use first 1000 images from training set

# Load the pre-trained YOLOv8 model
model = YOLO('/media/parashuram/dataset/YOLOv8_/results_75/runs/detect/train/weights/best.pt')

# Prepare the model for quantization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.model = model.model.to(device)
model.model.eval()

# Define quantization configuration
qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.model.qconfig = qconfig

# Prepare the model for static quantization
torch.quantization.prepare(model.model, inplace=True)

class CalibrationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        
        # Preprocess the image
        img = scale_image(img, (640, 640))  # Use a fixed size of 640x640
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255  # 0 - 255 to 0.0 - 1.0
        
        return img

def calibrate(model, dataset):
    image_paths = []
    for path in dataset:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        else:
            image_paths.append(path)
    
    cal_dataset = CalibrationDataset(image_paths)
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    print("Starting calibration...")
    for batch in cal_loader:
        batch = batch.to(device)
        model(batch)
    print("Calibration completed.")

print("Starting calibration...")
calibrate(model, calibration_dataset)

# Convert the model to quantized version
print("Converting model to quantized version...")
model.model = model.model.cpu()  # Move back to CPU for conversion
torch.quantization.convert(model.model, inplace=True)

# Save the quantized model
print("Saving quantized model...")
torch.save(model.state_dict(), 'yolov8_quantized.pt')

# Move the model to GPU for validation
model.to(device)

# Evaluate the quantized model
print(f"Evaluating quantized model on {device}...")
results = model.val(data=data['val'])  # Validate on the validation set

# Print accuracy metrics
print("Accuracy metrics:")
print(f"mAP: {results.box.map:.4f}")
print(f"mAP50: {results.box.map50:.4f}")
print(f"mAP75: {results.box.map75:.4f}")