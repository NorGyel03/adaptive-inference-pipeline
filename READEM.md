# Scene-Aware Adaptive Inference Pipeline

This repository implements a **scene-aware adaptive vision pipeline** for robotic waste segregation.

## Core Idea
- YOLO is used to detect and count objects in the workspace
- Scene complexity is determined by object count
- A lightweight or heavyweight classifier is selected dynamically

## Model Switching Rule
- **Objects ≤ 5** → Vision Transformer (ViT)
- **Objects > 5** → ResNet50

## Models (NOT included)
Due to size constraints, trained models are not included.

Expected files (place locally):
- `best0.pt` (YOLO detector)
- `resnet50_waste_classifier.onnx`
- `vit_waste_classifier.onnx`

## Run
```bash
pip install ultralytics opencv-python numpy onnxruntime
python adaptive_inference.py
