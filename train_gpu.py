from ultralytics import YOLO

if __name__ == '__main__':
    config_path = 'config.yaml'

    # model = YOLO('yolov10s.pt')   
    model = YOLO('weights/last.pt')

    model.train(
        data=config_path,
        epochs=5,
        imgsz=640, #old 384
        batch=8 ,#old 4
        workers=4,#old 4
        device='cuda',
        name='helicopter_detector_gpu'
    )
    print("Training completed.")
