from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
import torch
import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort #pip install deep-sort-realtime
from ultralytics import YOLO

# Pilih perangkat untuk eksekusi: CUDA jika tersedia, jika tidak gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Inception-v3 dengan weights terbaru
weights = Inception_V3_Weights.DEFAULT 
appearance_model = inception_v3(weights=weights).to(device)  
appearance_model.eval()

# Transformasi untuk preprocessing gambar sebelum memasukkan ke model appearance
transform = transforms.Compose([
    transforms.ToPILImage(),                        # Konversi frame dari format OpenCV (numpy) ke PIL Image
    transforms.Resize((224, 224)),                  # Resize gambar ke ukuran 224x224
    transforms.ToTensor(),                          # Konversi gambar ke format tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalisasi sesuai mean dan std ImageNet
                         std=[0.229, 0.224, 0.225]),
])

# Threshold jarak untuk menentukan apakah objek (misal botol Aqua) dekat dengan pergelangan tangan
threshold = 200

# Load YOLOv8 pose estimation model dan model deteksi Aqua
pose_model = YOLO('yolov8x-pose.pt').to(device)          # Model YOLOv8 untuk deteksi pose key points
aqua_model = YOLO('yolov8x.pt').to(device)     # Model YOLOv8 untuk deteksi objek Aqua (botol)

# Fungsi untuk mengecek apakah botol Aqua mendekati pergelangan tangan
def is_aqua_near_wrist(aqua_boxes, keypoints):
    if keypoints is None or len(keypoints) == 0:
        return False

    if len(keypoints) <= 10:  # Jika keypoints (titik tubuh) kurang dari 10, skip
        return False

    # Ambil keypoints pergelangan tangan kiri dan kanan
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    wrists = [left_wrist, right_wrist]

    for wrist in wrists:  # Loop melalui kedua pergelangan tangan
        wx, wy = wrist[0], wrist[1]
        if wx == 0 and wy == 0:  # Jika koordinat pergelangan tangan kosong, skip
            continue

        for aqua_box in aqua_boxes:  # Cek jarak setiap sudut kotak bounding box Aqua dengan pergelangan tangan
            x1, y1, x2, y2 = aqua_box[:4]
            box_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

            for corner in box_corners:
                distance = np.linalg.norm(np.array([wx, wy]) - np.array(corner))
                if distance < threshold:  # Jika jarak lebih kecil dari threshold, return True
                    return True

    return False

# Akses webcam dengan ID 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Inisialisasi DeepSORT Tracker untuk melacak objek secara berkelanjutan
tracker = DeepSort(max_age=100, n_init=3, nn_budget=150)

ret = True

while True:
    ret, frame = cap.read()  # Baca frame dari webcam

    if ret:
        # Deteksi pose dengan YOLOv8 pada frame
        pose_results = pose_model(frame, device=device)
        # Deteksi objek Aqua (botol) pada frame
        aqua_results = aqua_model(frame, device=device)

        # Simpan kotak bounding box dari deteksi Aqua
        aqua_boxes = []
        for r in aqua_results:
            for box in r.boxes:
                if box.cls == 39: 
                    aqua_boxes.append(box.xyxy[0].cpu().numpy())  # Simpan koordinat bounding box Aqua

                    # Menampilkan Jendela Open CV
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy()[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kotak biru untuk botol Aqua
                    cv2.putText(frame, 'Aqua', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # Label 'Aqua'

        # Ekstraksi keypoints (pose) jika tersedia dari hasil deteksi
        keypoints = None
        
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]  # Simpan keypoints
            detections = pose_results[0].boxes.xyxy.cpu().numpy()      # Bounding boxes
            confidences = pose_results[0].boxes.conf.cpu().numpy()     # Confidence score
            class_ids = pose_results[0].boxes.cls.cpu().numpy()        # Class IDs

            # Untuk pelacakan, siapkan list untuk deteksi dan fitur appearance (penampilan)
            detections_for_tracker = []
            appearance_features = []

            for box, conf, class_id in zip(detections, confidences, class_ids):
                if class_id == 0:  # Hanya proses class "person"
                    detections_for_tracker.append((box, conf, class_id))

                    # Ekstrak fitur penampilan dari bounding box "person"
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_person = frame[y1:y2, x1:x2]  # Crop bagian orang dari frame
                    if cropped_person.size > 0:
                        cropped_person_tensor = transform(cropped_person).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feature = appearance_model(cropped_person_tensor)  # Ekstrak fitur dari ResNet50
                        appearance_features.append(feature.cpu().numpy().flatten())  # Simpan sebagai array 1D
                    else:
                        appearance_features.append(np.zeros((512,)))  # Placeholder jika tidak ada fitur yang valid

            # Validasi dan pastikan semua fitur memiliki bentuk yang sama
            if len(appearance_features) > 0:
                appearance_features_array = np.array(appearance_features)
                if appearance_features_array.ndim == 2:
                    appearance_features_array = np.array([feat for feat in appearance_features_array if feat.size == 512])
                else:
                    appearance_features_array = np.zeros((len(detections_for_tracker), 512))  # Placeholder jika kosong

            # Update pelacak dengan deteksi dan fitur penampilan
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

            # Hubungkan ID pelacakan dengan fitur penampilan
            for i, track in enumerate(tracks):
                if track.is_confirmed() and i < len(appearance_features_array):
                    track.features = appearance_features_array[i].tolist()  # Simpan fitur sebagai list

            # Annotasi frame dengan ID pelacakan
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()  # Koordinat kotak pelacakan
                bbox = [int(i) for i in ltrb]
                cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Jika ada deteksi botol Aqua
        if aqua_boxes:
            print("Detected Aqua Boxes:", aqua_boxes)

        # Cek apakah botol Aqua dekat dengan pergelangan tangan
        is_near_wrist = is_aqua_near_wrist(aqua_boxes, keypoints)

        # Tambahkan teks status di frame yang menunjukkan status deteksi
        if is_near_wrist and keypoints is not None:
            # Cek ID orang yang mengambil Aqua
            taking_aqua_id = None
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(i) for i in ltrb]

                # Cek apakah bounding box orang overlap dengan bounding box Aqua
                for aqua_box in aqua_boxes:
                    x1, y1, x2, y2 = aqua_box[:4]
                    if (bbox[0] < x2 and bbox[2] > x1) and (bbox[1] < y2 and bbox[3] > y1):  # Cek overlap
                        taking_aqua_id = track_id
                        break  # Hentikan jika ditemukan

            # Menampilkan teks pada jendela OpenCV untuk deteksi ambil aqua
            if taking_aqua_id is not None:
                status_text = f"Orang dengan ID {taking_aqua_id} mengambil aqua"
                text_color = (0, 255, 0)  # Warna hijau
            else:
                status_text = "Tidak Ada Aqua yang diambil"
                text_color = (0, 0, 255)  # Warna merah
        else:
            status_text = "Tidak Ada Aqua yang diambil"
            text_color = (0, 0, 255)  # Warna merah

        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        # Plot hasil deteksi pose
        if len(pose_results) > 0:
            frame_pose = pose_results[0].plot()  # Gambar keypoints pada frame
        else:
            frame_pose = frame.copy()

        # Gabungkan frame pose dan frame asli
        combined_frame = cv2.addWeighted(frame_pose, 0.5, frame, 0.5, 0)

        # Tampilkan hasil di jendela OpenCV
        cv2.imshow('Webcam YOLOv8 Detection', combined_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Lepas akses webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
