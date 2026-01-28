import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = "best0.pt"
RESNET_ONNX_PATH = "resnet50_waste_classifier.onnx"
VIT_ONNX_PATH = "vit_waste_classifier.onnx"

SCENE_THRESHOLD = 5
INPUT_SIZE = 224

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
# CAMERA LOOP
# =========================
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count, model, results = adaptive_inference(frame)

        for x1, y1, x2, y2, cls in results:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Class {cls}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Objects: {count} | Model: {model}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("Adaptive Inference Pipeline", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
