from ultralytics import YOLO
import torch
import torch.quantization
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.ops import scale_image
from PIL import Image
import numpy as np
import os

# Load your dataset
data = check_det_dataset('/media/parashuram/dataset/YOLOv8_/datasets//data.yaml')

# Create a subset for calibration
calibration_dataset = data['train'][:1000]  # Use first 1000 images from training set

# Load the pre-trained YOLOv8 model
model = YOLO('/media/parashuram/dataset/YOLOv8_/results_75/runs/detect/train/weights/best.pt')

# Prepare the model for quantization
model.model = model.model.to('cpu')
model.model.eval()

# Define quantization configuration
qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.model.qconfig = qconfig

# Prepare the model for static quantization
torch.quantization.prepare(model.model, inplace=True)

# Calibrate the model
def calibrate(model, dataset):
    for path in dataset:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(root, file)
                        process_image(model, img_path)
        else:
            process_image(model, path)

def process_image(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    
    # Preprocess the image
    img = scale_image(img, (640, 640))  # Use a fixed size of 640x640
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    model.model(img)

calibrate(model, calibration_dataset)

# Convert the model to quantized version
torch.quantization.convert(model.model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), 'yolov8_quantized.pt')

# Evaluate the quantized model
results = model.val(data=data['val'])  # Validate on the validation set

# Print accuracy metrics
print(results.box.map)    # Mean Average Precision (mAP)
print(results.box.map50)  # mAP at IoU 0.5
print(results.box.map75)  # mAP at IoU 0.75