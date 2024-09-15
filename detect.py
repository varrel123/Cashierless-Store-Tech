import cv2
from ultralytics import YOLO

# Kalo mau train data run ini di CMD :
# yolo task=detect mode=train model=yolov8s.pt data=config.yaml epochs=100 imgsz=640 batch=4

# Sesuaikan Weight dengan ini : 

# model = YOLO("yolov8s.pt") 

model = YOLO("runs/detect/train4/weights/last.pt") 

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model(frame)

    detections = results[0]

    annotated_frame = detections.plot()

    for i, (box, score, class_id) in enumerate(zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls)):

        label = f"ID: {i} {model.names[int(class_id)]} {score:.2f}"

        x1, y1, x2, y2 = map(int, box)

        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Webcam YOLO Detection with IDs', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
