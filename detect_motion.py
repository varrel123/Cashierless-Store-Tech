import cv2
from ultralytics import YOLO
import torch

# Pastikan model menggunakan CUDA jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv8-pose model (yolov8l-pose in this case) dan pindahkan model ke device GPU (CUDA) jika tersedia
model = YOLO('yolov8x-pose.pt').to(device)

# Buka Webcam (index 0 biasanya untuk webcam default)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Lakukan inferensi menggunakan YOLOv8 pose
    results = model(frame)

    # Mendapatkan keypoints dari hasil deteksi
    for result in results:
        keypoints = result.keypoints.xy  # Mendapatkan koordinat keypoints
        conf = result.keypoints.conf     # Mendapatkan confidence score dari keypoints

        # Tampilkan keypoints yang terdeteksi dengan confidence di atas threshold
        threshold = 0.5  # Threshold untuk confidence
        print("Detected Keypoints:")
        for i, (kp, c) in enumerate(zip(keypoints[0], conf[0])):
            if c > threshold:
                print(f"Keypoint {i}: Position (x={kp[0]}, y={kp[1]}) with confidence {c}")

    # Menggambar keypoints pada frame yang sedang ditampilkan
    annotated_frame = results[0].plot()

    # Tampilkan hasil deteksi pada layar
    cv2.imshow('Pose Detection', annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah loop selesai, lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
