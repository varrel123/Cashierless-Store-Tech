from ultralytics import YOLO
import torch

def main():
    # Check if CUDA is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the model and move it to the GPU
    model = YOLO("yolov8m.pt")

    model.to(device)

    # Train the model
    model.train(
        data='config.yaml', 
        epochs=100, 
        imgsz=640, 
        batch=4, 
        device=device, 
        amp=False  
    )

if __name__ == '__main__':
    main()

# https://colab.research.google.com/drive/1ZPeIiKpmUETtnUPFYQsA7qlUBcw3gHGO#scrollTo=Ef-4YQtewvFa

