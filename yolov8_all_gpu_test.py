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

# Define updated quantization configuration
def get_qconfig(is_per_channel=False):
    if is_per_channel:
        return torch.quantization.QConfig(
            activation=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8
            ),
            weight=torch.quantization.observer.PerChannelMinMaxObserver.with_args(
                quant_min=-128, quant_max=127, dtype=torch.qint8
            )
        )
    else:
        return torch.quantization.QConfig(
            activation=torch.quantization.observer.MovingAverageMinMaxObserver.with_args(
                quant_min=0, quant_max=255, dtype=torch.quint8
            ),
            weight=torch.quantization.observer.MinMaxObserver.with_args(
                quant_min=-128, quant_max=127, dtype=torch.qint8
            )
        )

qconfig = get_qconfig(is_per_channel=False)
model.model.qconfig = qconfig

# Prepare the model for static quantization
torch.quantization.prepare(model.model, inplace=True)

class CalibrationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = [path for path in image_paths if os.path.exists(path)]
        print(f"Total valid images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            # Preprocess the image
            img = scale_image(img, (640, 640))  # Use a fixed size of 640x640
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).float()
            img /= 255  # 0 - 255 to 0.0 - 1.0
            
            return img
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            return torch.zeros((3, 640, 640))  # Return a blank image in case of error

def get_image_paths(dataset):
    image_paths = []
    for path in dataset:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        if os.path.exists(full_path):
                            image_paths.append(full_path)
        elif os.path.exists(path):
            image_paths.append(path)
    return image_paths

def calibrate(model, dataset):
    image_paths = get_image_paths(dataset)
    print(f"Total images found: {len(image_paths)}")
    
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