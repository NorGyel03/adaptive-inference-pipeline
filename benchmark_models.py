import time
import psutil
import numpy as np
import onnxruntime as ort

INPUT_SIZE = 224
RUNS = 30   # average over multiple runs

RESNET_MODEL = "resnet50_waste_classifier.onnx"
VIT_MODEL = "vit_waste_classifier.onnx"

process = psutil.Process()

def preprocess():
    img = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def benchmark(model_path, name):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    img = preprocess()

    cpu_before = process.cpu_percent(interval=1)

    start = time.time()
    for _ in range(RUNS):
        session.run(None, {input_name: img})
    end = time.time()

    cpu_after = process.cpu_percent(interval=1)

    avg_latency = ((end - start) / RUNS) * 1000
    avg_cpu = (cpu_before + cpu_after) / 2

    print(f"\n{name}")
    print(f"Average inference latency: {avg_latency:.2f} ms")
    print(f"Average CPU usage: {avg_cpu:.2f} %")

# =========================
# RUN BENCHMARKS
# =========================
benchmark(RESNET_MODEL, "ResNet50 (Lightweight)")
benchmark(VIT_MODEL, "ViT (Heavyweight)")
