import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r'C:\Users\Sailappan\Desktop\Liveness\runs\detect\train7\weights\best.pt'  # Path to your trained model
CONFIDENCE_THRESHOLD = 0.5  # High noise threshold
PLC_TRIGGER_ACTIVE = False  # Simulated PLC Trigger (Constraint #4)

# Load your best.pt model
model = YOLO(MODEL_PATH)

# Initialize Camera (Index 0 is usually the default monochromatic/webcam)
cap = cv2.VideoCapture(0)

print("Starting Fabric Defect Detection System...")
print("Press 't' to toggle PLC Trigger (Simulate fabric folding)")
print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Grayscale Preprocessing (Matching your training constraint)
    # Even if the camera is mono, OpenCV often reads in BGR format
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert back to 3-channel gray for YOLO compatibility
    display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # 2. Constraint #4: Only run model when fabric is folding
    if PLC_TRIGGER_ACTIVE:
        results = model.predict(display_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # 3. Constraint #5: Extract Coordinates for Stamper
        for r in results:
            for box in r.boxes:
                # Get Center Coordinates (x, y)
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                label = f"DEFECT at ({center_x}, {center_y})"
                
                # Visual Feedback
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.circle(display_frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, label, (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Here you would send (center_x, center_y) to your PLC via Modbus/Serial
                # print(f"STAMP_TRIGGER: X={center_x}, Y={center_y}")

    # Display Status
    status_color = (0, 255, 0) if PLC_TRIGGER_ACTIVE else (0, 0, 255)
    status_text = "FOLDING ACTIVE" if PLC_TRIGGER_ACTIVE else "IDLE - WAITING FOR PLC"
    cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    cv2.imshow("Fabric Defect Monitor", display_frame)

    # Keyboard Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        PLC_TRIGGER_ACTIVE = not PLC_TRIGGER_ACTIVE

cap.release()
cv2.destroyAllWindows()