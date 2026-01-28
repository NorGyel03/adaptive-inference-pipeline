import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "resnet50_waste_classifier.onnx"
INPUT_SIZE = 224

# Load model
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Dummy image
img = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

# Preprocess
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)

# Inference
start = time.time()
output = session.run(None, {input_name: img})
end = time.time()

print("Output shape:", output[0].shape)
print("Inference time (ms):", round((end - start) * 1000, 2))
