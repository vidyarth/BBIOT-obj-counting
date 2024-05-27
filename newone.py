from ultralytics import YOLO

# Path to your best.pt model file
model_path = "best.pt"

# Load the YOLOv8 model
model = YOLO(model_path)

# Optional: Set confidence threshold (default 0.5)
model.conf = 0.6  # Adjust this value as needed

# Optional: Set class names list (if your model has custom classes)
# class_names = ["person", "car", "bicycle"]  # Replace with your class names
# model.names = class_names

# Start video capture (replace 0 for external camera)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Run inference on the frame
    results = model(frame)

    # Display the results (optional)
    results.render()  # This will display the frame with bounding boxes and labels
    cv2.imshow("YOLOv8 Detection", frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
