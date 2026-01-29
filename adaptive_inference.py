import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import serial
import struct

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "best.pt"
RESNET_ONNX_PATH = "resnet50_waste_classifier.onnx"
VIT_ONNX_PATH = "vit_waste_classifier.onnx"

SERIAL_PORT = "COM3"      # CHANGE if needed
BAUD_RATE = 2_000_000

SCENE_THRESHOLD = 2
INPUT_SIZE = 224

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

YOLO_ANALYSIS_TIME = 10.0     # seconds
CLASSIFICATION_HOLD_TIME = 10.0

live_object_count = 0        # updated every YOLO frame
frozen_object_count = None  # latched value

analysis_boxes = []   # stores all YOLO boxes for ANALYSIS display


import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


def get_system_stats():
    cpu = psutil.cpu_percent(interval=None)

    gpu = None
    if GPU_AVAILABLE:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        gpu = util.gpu

    return cpu, gpu

import time

state = "ANALYSIS"
state_start_time = time.time()

frozen_object_count = 0
best_bbox = None
best_conf = 0.0
classified_label = None
model_used = None

# =========================
# LOAD MODELS
# =========================
print("[INFO] Loading models...")

yolo = YOLO(YOLO_MODEL_PATH)

resnet_sess = ort.InferenceSession(
    RESNET_ONNX_PATH, providers=["CPUExecutionProvider"]
)

vit_sess = ort.InferenceSession(
    VIT_ONNX_PATH, providers=["CPUExecutionProvider"]
)

print("[INFO] Models loaded successfully")

# =========================
# SERIAL CAMERA INIT
# =========================
print("[INFO] Connecting to ESP32 camera...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
print("[INFO] ESP32 camera connected")

# =========================
# PREPROCESS
# =========================
def preprocess_crop(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# CLASSIFIERS
# =========================
def classify_resnet(crop):
    inp = preprocess_crop(crop)
    input_name = resnet_sess.get_inputs()[0].name
    output = resnet_sess.run(None, {input_name: inp})[0]
    return int(np.argmax(output))

def classify_vit(crop):
    inp = preprocess_crop(crop)
    input_name = vit_sess.get_inputs()[0].name
    output = vit_sess.run(None, {input_name: inp})[0]
    return int(np.argmax(output))

# =========================
# ADAPTIVE INFERENCE
# =========================
def adaptive_inference(frame):
    detections = yolo(frame, conf=0.4, verbose=False)[0]

    boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
    object_count = len(boxes)

    if object_count <= SCENE_THRESHOLD:
        classifier = classify_vit
        model_used = "ViT (Heavy)"
    else:
        classifier = classify_resnet
        model_used = "ResNet50 (Light)"

    results = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        cls = classifier(crop)
        results.append((x1, y1, x2, y2, cls))

    return object_count, model_used, results

# =========================
# ESP32 FRAME READER
# =========================
def read_esp32_frame():
    # Wait for JPEG start marker
    while True:
        if ser.read(2) == b'\xff\xd8':
            break

    size_bytes = ser.read(4)
    if len(size_bytes) != 4:
        return None

    size = struct.unpack('<I', size_bytes)[0]
    jpg = ser.read(size)

    if len(jpg) != size:
        return None

    img = cv2.imdecode(
        np.frombuffer(jpg, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    return img

# =========================
# MAIN LOOP
# =========================
def main():
    global state, state_start_time
    global frozen_object_count, best_bbox, best_conf
    global classified_label, model_used

    cv2.namedWindow("ESP32 Adaptive Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ESP32 Adaptive Inference", WINDOW_WIDTH, WINDOW_HEIGHT)

    while True:
        frame = read_esp32_frame()
        if frame is None:
            continue

        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        now = time.time()

        # =========================
        # PHASE 1: YOLO ANALYSIS
        # =========================
        # =========================
        # STATE MACHINE
        # =========================
        if state == "ANALYSIS":
            detections = yolo(frame, conf=0.4, verbose=False)[0]

            analysis_boxes = detections.boxes.xyxy.cpu().numpy() if detections.boxes else []
            confs = detections.boxes.conf.cpu().numpy() if detections.boxes else []

            live_object_count = len(analysis_boxes)

            # Track highest confidence detection
            for box, conf in zip(analysis_boxes, confs):
                if conf > best_conf:
                    best_conf = conf
                    best_bbox = box.astype(int)

            # ----- TRANSITION: ANALYSIS → FREEZE -----
            if now - state_start_time > YOLO_ANALYSIS_TIME:
                state = "CLASSIFY"
                state_start_time = now

                frozen_object_count = live_object_count
                analysis_boxes = [] 

                # Choose classifier ONCE
                if frozen_object_count <= SCENE_THRESHOLD:
                    model_used = "ViT"
                    classifier = classify_vit
                else:
                    model_used = "ResNet50"
                    classifier = classify_resnet

                # Run classification ONCE
                if best_bbox is not None:
                    x1, y1, x2, y2 = best_bbox
                    crop = frame[y1:y2, x1:x2]
                    classified_label = classifier(crop)

        # =========================
        # FREEZE / CLASSIFY STATE
        # =========================
        else:
            # ----- TRANSITION: FREEZE → ANALYSIS -----
            if now - state_start_time > CLASSIFICATION_HOLD_TIME:
                state = "ANALYSIS"
                state_start_time = now

                # Reset only what must be recomputed
                best_conf = 0.0
                best_bbox = None
                classified_label = None
                frozen_object_count = None


        # =========================
# DRAWING
# =========================

        # ---- ANALYSIS: draw ALL detected boxes ----
        if state == "ANALYSIS":
            for box in analysis_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ---- CLASSIFY / FREEZE: draw ONLY best box ----
        elif state == "CLASSIFY" and best_bbox is not None:
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            if classified_label is not None:
                cv2.putText(
                    frame,
                    f"Class {classified_label}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # =========================
        # SYSTEM OVERLAY
        # =========================
        cpu, gpu = get_system_stats()

        cv2.putText(
            frame, f"State: {state}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )

        display_count = (
            live_object_count if state == "ANALYSIS"
            else frozen_object_count
        )

        cv2.putText(
            frame, f"Objects: {display_count}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"Model: {model_used}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame, f"CPU: {cpu:.1f}%", (480, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )

        if gpu is not None:
            cv2.putText(
                frame, f"GPU: {gpu}%", (480, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
            )

        cv2.imshow("ESP32 Adaptive Inference", frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break



# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
