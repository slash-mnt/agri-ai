from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',
        epochs=1200,
        imgsz=576,
        device='cuda',
        batch=-1
    )

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
