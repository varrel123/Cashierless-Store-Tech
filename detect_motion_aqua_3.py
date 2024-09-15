from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Select device: CUDA if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
threshold = 300  # Increase threshold

# Load YOLOv8 pose estimation model and object detection model
pose_model = YOLO('yolov8x-pose.pt').to(device)
aqua_model = YOLO('best_aqua_bram_1.pt').to(device)

# Adjust confidence threshold for Aqua detection
aqua_conf_threshold = 0.5  # You can adjust this value

# Function to check if Aqua bottle is held by wrist
def is_aqua_held_by_wrist(aqua_boxes, keypoints, threshold=300):
    if keypoints is None or len(keypoints) == 0:
        return False, None

    for person_idx, person_keypoints in enumerate(keypoints):
        if len(person_keypoints) <= 10:
            continue

        left_wrist = person_keypoints[9]
        right_wrist = person_keypoints[10]
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
                        return True, person_idx

    return False, None

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

        keypoints = None
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints.xy.cpu().numpy()  # Get all people

        if aqua_boxes:
            print("Detected Aqua Boxes:", aqua_boxes)
        else:
            print("No Aqua Boxes Detected.")

        # Check if Aqua bottle is held by wrist
        is_held_by_wrist, person_holding_idx = is_aqua_held_by_wrist(aqua_boxes, keypoints, threshold)

        # Draw bounding boxes and labels for each person
        for idx, result in enumerate(pose_results[0].boxes):
            box = result.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box[:4].astype(int)
            color = (0, 255, 0) if idx == person_holding_idx else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Person {idx+1}"
            if idx == person_holding_idx:
                label += " (Holding Aqua)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Plot results for pose and Aqua
        frame_pose = pose_results[0].plot()
        frame_aqua = aqua_results[0].plot()

        # Combine both detection results into the same frame
        combined_frame = cv2.addWeighted(frame_pose, 0.5, frame_aqua, 0.5, 0)

        # Display the result in OpenCV window
        cv2.imshow('Webcam YOLOv8 Detection', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()