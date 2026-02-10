import cv2
from ultralytics import YOLO
import time

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'      # Path to your trained model
CAMERA_INDEX = 0            # 0 is usually the default USB camera. Try 1 if 0 fails.
CONFIDENCE_THRESHOLD = 0.45 # Set lower (0.45) to catch more defects (Recall > Precision)
                            # Set higher (0.60) to avoid false alarms

# --- 1. LOAD THE MODEL ---
print("Loading AI Model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 2. CONNECT TO CAMERA ---
print("Connecting to Camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)

# Optimize Camera for Industry (Optional but recommended)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Error: Could not open camera. Check connection.")
    exit()

print("✅ Camera connected! Press 'q' to quit.")

# --- 3. MAIN LOOP ---
while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to read frame from camera.")
        break

    # Run Inference (The AI Brain)
    # stream=True makes it run faster for video
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Check results
    defect_detected = False
    
    for r in results:
        # Draw the boxes on the frame
        annotated_frame = r.plot()
        
        # Check if any boxes were found
        if len(r.boxes) > 0:
            defect_detected = True
            
            # --- SIMULATED PLC SIGNAL ---
            # In the real version, this is where we send the Modbus signal
            cv2.putText(annotated_frame, "!!! DEFECT DETECTED !!!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(annotated_frame, "SIGNAL: STOP MACHINE", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Print to console for debugging
            print(f"🔴 DEFECT FOUND! Confidence: {r.boxes.conf[0]:.2f}")

    # Display the output
    if defect_detected:
        # Show defective frames with a red border
        cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
        cv2.imshow('Fabric Defect Detection', annotated_frame)
    else:
        # Show clean frames normally
        cv2.imshow('Fabric Defect Detection', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("Program stopped.")