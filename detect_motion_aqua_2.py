from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time

# Select device: CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
threshold = 100  # Increase threshold

# Load YOLOv8 pose estimation model and object detection model
pose_model = YOLO('yolov8x-pose.pt').to(device)
aqua_model = YOLO('best_aqua_bram_1.pt').to(device)

# Adjust confidence threshold for Aqua detection
aqua_conf_threshold = 0.3  # You can adjust this value

# Define keypoint names
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Function to check if Aqua bottle is held by wrist
def is_aqua_held_by_wrist(aqua_boxes, keypoints, threshold=50):  # Increase threshold
    if keypoints is None or len(keypoints) == 0:
        print("No keypoints detected.")
        return False

    print(f"Pose Keypoints Detected: {keypoints}")

    if len(keypoints) <= 10:
        print("Not enough keypoints detected.")
        return False

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    wrists = [left_wrist, right_wrist]

    for wrist in wrists:
        wx, wy = wrist[0], wrist[1]  # Access coordinates directly
        if wx == 0 and wy == 0:  # Skip if wrist coordinates are (0, 0)
            continue
        print(f"Wrist coordinates: ({wx}, {wy})")

        for aqua_box in aqua_boxes:
            x1, y1, x2, y2 = aqua_box[:4]
            box_corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            
            for corner in box_corners:
                distance = np.linalg.norm(np.array([wx, wy]) - np.array(corner))
                print(f"Distance from wrist to box corner {corner}: {distance}")
                print(f"Aqua Box Coordinates: {aqua_box}")

                if distance < threshold:
                    print("Aqua bottle is held by the wrist.")
                    return True

    print("Aqua bottle is not held by the wrist.")
    return False

# Access webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret = True

# Loop to read frames from webcam
while ret:
    ret, frame = cap.read()

    if ret:
        img_height, img_width = frame.shape[:2]

        # Detect pose from person in frame using pose model
        pose_results = pose_model(frame, device=device)
        # Detect Aqua objects in frame using aqua model
        aqua_results = aqua_model(frame, conf=aqua_conf_threshold, device=device)

        # Get bounding boxes from Aqua detection results
        aqua_boxes = []
        for r in aqua_results:
            for box in r.boxes:
                aqua_boxes.append(box.xyxy[0].cpu().numpy())
                print(f"Detected Aqua Box: {box.xyxy[0].cpu().numpy()}, Confidence: {box.conf.item():.2f}")

        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()[0]  # Use .xy to get coordinates

            # Draw all keypoints and label them
            for i, keypoint in enumerate(keypoints):
                x, y = int(keypoint[0]), int(keypoint[1])  # Extract coordinates properly
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Keypoint
                cv2.putText(frame, keypoint_names[i], (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Check if wrist keypoints are available
            if len(keypoints) > 10:
                left_wrist = keypoints[9][:2]
                right_wrist = keypoints[10][:2]
                
                # Log the coordinates of the wrist keypoints
                if not (left_wrist[0] == 0 and left_wrist[1] == 0):
                    print(f"Left Wrist Coordinates: {left_wrist}")
                if not (right_wrist[0] == 0 and right_wrist[1] == 0):
                    print(f"Right Wrist Coordinates: {right_wrist}")
                
                # Draw wrist circles
                for wrist in [left_wrist, right_wrist]:
                    wx, wy = int(wrist[0]), int(wrist[1])
                    if wx == 0 and wy == 0:  # Skip if wrist coordinates are (0, 0)
                        continue
                    cv2.circle(frame, (wx, wy), threshold, (255, 0, 0), 2)  # Radius circle
            else:
                print("Wrist keypoints not detected.")
        else:
            print("No keypoints detected.")
            keypoints = None

        if aqua_boxes:
            print("Detected Aqua Boxes:", aqua_boxes)
        else:
            print("No Aqua Boxes Detected.")

        # Check if Aqua bottle is held by wrist
        is_held_by_wrist = is_aqua_held_by_wrist(aqua_boxes, keypoints, threshold)

        status_text = "Aqua bottle is held by the wrist!" if is_held_by_wrist else "Aqua bottle is not held by the wrist."
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_held_by_wrist else (0, 0, 255), 2, cv2.LINE_AA)

        for aqua_box in aqua_boxes:
            x1, y1, x2, y2 = map(int, aqua_box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('Webcam YOLOv8 Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # time.sleep(1)

cap.release()
cv2.destroyAllWindows()