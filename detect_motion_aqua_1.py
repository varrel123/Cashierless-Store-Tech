from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Pilih perangkat untuk eksekusi: CUDA jika tersedia, jika tidak gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

threshold = 100

# Load YOLOv8 pose estimation model dan model deteksi Aqua
pose_model = YOLO('yolov8x-pose.pt').to(device)
aqua_model = YOLO('best_aqua_bram_1.pt').to(device)

# Fungsi untuk mengecek apakah botol Aqua mendekati pergelangan tangan
def is_aqua_near_wrist(aqua_boxes, keypoints, threshold=300):
    if keypoints is None or len(keypoints) == 0:
        print("No keypoints detected.")
        return False

    print(f"Pose Keypoints Detected: {keypoints}")

    # Cek apakah keypoints memiliki data valid
    if len(keypoints) <= 10:
        print("Not enough keypoints detected.")
        return False

    # Ambil keypoints untuk pergelangan tangan
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    wrists = [left_wrist, right_wrist]

    for wrist in wrists:
        wx, wy = wrist[0], wrist[1]
        if wx == 0 and wy == 0:  
            continue
        print(f"Wrist coordinates: ({wx}, {wy})")

        for aqua_box in aqua_boxes:
            x1, y1, x2, y2 = aqua_box[:4]
            box_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

            for corner in box_corners:
                distance = np.linalg.norm(np.array([wx, wy]) - np.array(corner))
                print(f"Distance from wrist to box corner {corner}: {distance}")
                print(f"Aqua Box Coordinates: {aqua_box}")

            # Periksa apakah pergelangan tangan berada dekat dengan pusat bounding box Aqua
            if distance < threshold:
                print("Aqua bottle is held by the wrist.")
                return True
    
    print("Aqua bottle is not held by the wrist.")
    return False

# Akses webcam 0
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret = True

# Loop untuk membaca frame dari webcam
while ret:
    ret, frame = cap.read()

    if ret:
        img_height, img_width = frame.shape[:2]  # Ambil dimensi gambar

        # Deteksi pose dari person di frame menggunakan pose model
        pose_results = pose_model(frame, device=device)
        # Deteksi objek Aqua di frame menggunakan aqua model
        aqua_results = aqua_model(frame, device=device)

        # Get bounding boxes from Aqua detection results
        aqua_boxes = []
        for r in aqua_results:
            for box in r.boxes:
                aqua_boxes.append(box.xyxy[0].cpu().numpy())
                print(f"Detected Aqua Box: {box.xyxy[0].cpu().numpy()}, Confidence: {box.conf.item():.2f}")


        # Cek hasil deteksi pose
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]  # Use .xy to get coordinates

        # Cek jika bounding box Aqua terdeteksi
        if aqua_boxes:
            print("Detected Aqua Boxes:", aqua_boxes)
        else:
            print("No Aqua Boxes Detected.")

        # Cek apakah botol Aqua berada dekat dengan pergelangan tangan
        is_near_wrist = is_aqua_near_wrist(aqua_boxes, keypoints,threshold)

        # Tambahkan teks di frame yang menunjukkan status deteksi
        status_text = "Aqua bottle is near the wrist!" if is_near_wrist else "Aqua bottle is not near the wrist."
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_near_wrist else (0, 0, 255), 2, cv2.LINE_AA)

        # Plot hasil deteksi untuk pose dan Aqua
        frame_pose = pose_results[0].plot()
        frame_aqua = aqua_results[0].plot()

        # Gabungkan kedua hasil deteksi ke frame yang sama
        combined_frame = cv2.addWeighted(frame_pose, 0.5, frame_aqua, 0.5, 0)

        # Visualisasikan bounding box dan keypoints
        for aqua_box in aqua_boxes:
            x1, y1, x2, y2 = map(int, aqua_box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if len(keypoints) > 10:
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
                
            cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 255, 0), -1)

        # Tampilkan hasil di jendela OpenCV
        cv2.imshow('Webcam YOLOv8 Detection', combined_frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Lepas akses webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
