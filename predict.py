import cv2
import os
from ultralytics import YOLO

# Load the trained model (replace with your trained model path if different)
model_path = 'test_weights/weights/last.pt'  # or 'yolov10x.pt' for pretrained
if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Using pretrained model.")
    model_path = 'yolov10x.pt'

model = YOLO(model_path)

# Path to the video file (replace with your video path)
video_path = 'h1.mp4'  # Example: use one of your videos like 'backup/h1.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Resize factor (e.g., 0.5 for half size, 2.0 for double size)
resize_factor = 0.70  # Change this value to resize the display window

# Calculate new dimensions
new_width = int(width * resize_factor)
new_height = int(height * resize_factor)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, conf=0.5, show=False, verbose=False)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Resize the annotated frame for display
    resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow('YOLO Detection', resized_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Prediction completed.")
