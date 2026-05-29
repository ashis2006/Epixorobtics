from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best (5).pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Plastic Bottle Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()