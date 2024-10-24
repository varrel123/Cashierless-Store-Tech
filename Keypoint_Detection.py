import cv2
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Pastikan model menggunakan CUDA jika tersedia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the YOLOv8-pose model (yolov8x-pose in this case) dan pindahkan model ke device GPU (CUDA) jika tersedia
model = YOLO('yolov8x-pose.pt')

# Buka Webcam (index 0 biasanya untuk webcam default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize DeepSORT Tracker
tracker = DeepSort(max_age=80, n_init=3, nn_budget=70)  # max_age dinaikkan agar ID bertahan lebih lama

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Lakukan inferensi menggunakan YOLOv8 pose
    results = model.predict(frame, device=device)

    # Parse hasil deteksi
    detections = results[0].boxes.xyxy.cpu().numpy()  # Dapatkan koordinat bbox [x1, y1, x2, y2]
    confidences = results[0].boxes.conf.cpu().numpy()  # Dapatkan confidence score
    class_ids = results[0].boxes.cls.cpu().numpy()  # Dapatkan class ID (untuk orang, ID biasanya = 0)

    # Gabungkan data menjadi format deteksi untuk tracker
    detections_for_tracker = []
    for i, (box, conf, class_id) in enumerate(zip(detections, confidences, class_ids)):
        if class_id == 0:  # Hanya proses class "person"
            detections_for_tracker.append((box, conf, class_id))

    # Update pelacak (tracker) menggunakan deteksi yang ada
    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # Annotasi frame dengan ID pelacakan (tanpa menggambar bounding box tambahan)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom bounding box
        bbox = [int(i) for i in ltrb]

        # Tambahkan ID ke bounding box YOLOv8-pose
        cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Annotasi frame dengan hasil deteksi pose
    annotated_frame = results[0].plot()  # Plot hasil deteksi pose pada frame

    # Gabungkan frame dengan bounding box pelacakan dan pose
    combined_frame = cv2.addWeighted(annotated_frame, 0.7, frame, 0.3, 0)

    # Tampilkan hasil deteksi pada layar
    cv2.imshow('Pose Detection with Tracking', combined_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah loop selesai, lepaskan kamera dan tutup semua jendela OpenCV
cap.release()
cv2.destroyAllWindows()
