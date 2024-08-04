from ultralytics import YOLO
import torch
import torch.quantization

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # or use your custom trained model

# Prepare the model for quantization
model.model = model.model.to('cpu')
model.model.eval()

# Define quantization configuration
qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model.model, inplace=True)

# Calibrate the model (you need a representative dataset)
def calibrate(model, dataset):
    for data in dataset:
        model(data)

# Assuming you have a calibration dataset
calibration_dataset = [...]  # Your calibration dataset
calibrate(model.model, calibration_dataset)

# Convert the model to quantized version
torch.quantization.convert(model.model, inplace=True)

# Save the quantized model
torch.save(model.state_dict(), 'yolov8_quantized.pt')

# Evaluate the quantized model
results = model.val()  # Validate on the default dataset or specify your own

# Print accuracy metrics
print(results.box.map)    # Mean Average Precision (mAP)
print(results.box.map50)  # mAP at IoU 0.5
print(results.box.map75)  # mAP at IoU 0.75