from ultralytics import YOLO
import cv2
import torch

# Pilih perangkat untuk eksekusi: CUDA jika tersedia, jika tidak gunakan CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model dan pindahkan ke perangkat yang dipilih
model = YOLO('best_indomie_bram.pt').to(device)

# Akses webcam 0
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret = True

# Loop untuk membaca frame dari webcam
while ret:
    ret, frame = cap.read()

    if ret:
        # Deteksi dan lacak objek, kirim frame ke GPU jika menggunakan CUDA
        results = model.track(frame, persist=True, device=device)

        # Plot hasil deteksi pada frame
        frame_ = results[0].plot()

        # Visualisasi hasil
        cv2.imshow('Webcam YOLOv8 Detection', frame_)
        
        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Lepas akses webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
