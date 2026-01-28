import onnxruntime as ort
from ultralytics import YOLO

# Paths
YOLO_MODEL = "best0.pt"
RESNET_MODEL = "resnet50_waste_classifier.onnx"
VIT_MODEL = "vit_waste_classifier.onnx"

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)
print("YOLO loaded ")

print("Loading ResNet50 ONNX...")
resnet = ort.InferenceSession(RESNET_MODEL, providers=["CPUExecutionProvider"])
print("ResNet50 loaded ")

print("Loading ViT ONNX...")
vit = ort.InferenceSession(VIT_MODEL, providers=["CPUExecutionProvider"])
print("ViT loaded ")

print("\nALL MODELS LOADED SUCCESSFULLY ")
