from ultralytics import YOLO
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort  
from torchvision import models, transforms


# Pilih perangkat untuk eksekusi: CUDA jika tersedia, jika tidak gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load model untuk fitur penampilan
appearance_model = models.resnet50(pretrained=True).to(device)
appearance_model.eval()

# Transformasi untuk preprocessing gambar
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

threshold = 100

# Load YOLOv8 pose estimation model dan model deteksi Aqua
pose_model = YOLO('yolov8x-pose.pt').to(device)
aqua_model = YOLO('best_aqua_bram_1.pt').to(device)

# Fungsi untuk mengecek apakah botol Aqua mendekati pergelangan tangan
def is_aqua_near_wrist(aqua_boxes, keypoints, threshold=300):
    if keypoints is None or len(keypoints) == 0:
        return False

    if len(keypoints) <= 10:
        return False

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    wrists = [left_wrist, right_wrist]

    for wrist in wrists:
        wx, wy = wrist[0], wrist[1]
        if wx == 0 and wy == 0:  
            continue

        for aqua_box in aqua_boxes:
            x1, y1, x2, y2 = aqua_box[:4]
            box_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

            for corner in box_corners:
                distance = np.linalg.norm(np.array([wx, wy]) - np.array(corner))
                if distance < threshold:
                    return True

    return False

# Akses webcam 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Inisialisasi DeepSORT Tracker
# Inisialisasi DeepSORT Tracker
tracker = DeepSort(max_age=100, n_init=3, nn_budget=150)

ret = True

while ret:
    ret, frame = cap.read()

    if ret:
        pose_results = pose_model(frame, device=device)
        aqua_results = aqua_model(frame, device=device)

        aqua_boxes = []
        for r in aqua_results:
            for box in r.boxes:
                aqua_boxes.append(box.xyxy[0].cpu().numpy())

        keypoints = None
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]
            detections = pose_results[0].boxes.xyxy.cpu().numpy()
            confidences = pose_results[0].boxes.conf.cpu().numpy()
            class_ids = pose_results[0].boxes.cls.cpu().numpy()

            detections_for_tracker = []
            appearance_features = []

            for box, conf, class_id in zip(detections, confidences, class_ids):
                if class_id == 0:  # Hanya proses class "person"
                    detections_for_tracker.append((box, conf, class_id))
                    # Ekstrak fitur penampilan
                    x1, y1, x2, y2 = map(int, box[:4])
                    cropped_person = frame[y1:y2, x1:x2]
                    if cropped_person.size > 0:
                        cropped_person_tensor = transform(cropped_person).unsqueeze(0).to(device)
                        with torch.no_grad():
                            feature = appearance_model(cropped_person_tensor)
                        appearance_features.append(feature.cpu().numpy().flatten())  # Pastikan array 1D
                    else:
                        appearance_features.append(np.zeros((512,)))  # Placeholder jika tidak ada

            # Pastikan semua fitur memiliki bentuk yang sama
            if len(appearance_features) > 0:
                appearance_features_array = np.array(appearance_features)
                # Jika fitur tidak memiliki dimensi yang seragam, buang fitur yang tidak valid
                if appearance_features_array.ndim == 2:
                    appearance_features_array = np.array([feat for feat in appearance_features_array if feat.size == 512])
                else:
                    appearance_features_array = np.zeros((len(detections_for_tracker), 512))

            # Update pelacak menggunakan deteksi
            tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

            # Hubungkan fitur penampilan dengan ID pelacakan
            for i, track in enumerate(tracks):
                if track.is_confirmed() and i < len(appearance_features_array):
                    track.features = appearance_features_array[i].tolist()  # Menyimpan fitur sebagai list

            # Annotasi frame dengan ID pelacakan
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(i) for i in ltrb]
                cv2.putText(frame, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Cek jika bounding box Aqua terdeteksi
        if aqua_boxes:
            print("Detected Aqua Boxes:", aqua_boxes)

        # Cek apakah botol Aqua berada dekat dengan pergelangan tangan
        is_near_wrist = is_aqua_near_wrist(aqua_boxes, keypoints, threshold)

        # Tambahkan teks di frame yang menunjukkan status deteksi
        if is_near_wrist and keypoints is not None:
            # Cek ID orang yang mengambil Aqua
            taking_aqua_id = None
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bbox = [int(i) for i in ltrb]

                # Cek jika bounding box orang dekat dengan pergelangan tangan
                for aqua_box in aqua_boxes:
                    x1, y1, x2, y2 = aqua_box[:4]
                    if (bbox[0] < x2 and bbox[2] > x1) and (bbox[1] < y2 and bbox[3] > y1):  # Cek overlap
                        taking_aqua_id = track_id
                        break  # Hentikan setelah menemukan ID

            # Atur status_text sesuai dengan ID yang ditemukan
            if taking_aqua_id is not None:
                status_text = f"Orang dengan ID {taking_aqua_id} mengambil aqua"
                text_color = (0, 255, 0)  # Hijau
            else:
                status_text = "Tidak Ada Aqua yang diambil"
                text_color = (0, 0, 255)  # Merah
        else:
            status_text = "Tidak Ada Aqua yang diambil"
            text_color = (0, 0, 255)  # Merah

        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)


        # Plot hasil deteksi untuk pose
        if len(pose_results) > 0:
            frame_pose = pose_results[0].plot()
        else:
            frame_pose = frame.copy()

        # Gabungkan frame yang telah diproses
        combined_frame = cv2.addWeighted(frame_pose, 0.5, frame, 0.5, 0)

        # Tampilkan hasil di jendela OpenCV
        cv2.imshow('Webcam YOLOv8 Detection', combined_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Lepas akses webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
