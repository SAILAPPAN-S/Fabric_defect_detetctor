import cv2
from ultralytics import YOLO
import numpy as np

# --- CONFIGURATION: ULTRA-STRICT HOLE-ONLY ---
MODEL_PATH = 'notebook/best.pt'
CONFIDENCE_THRESHOLD = 0.50      # Lowered slightly because filters are strict
MIN_DEFECT_AREA = 3000           # Ignore tiny noise
MAX_DEFECT_AREA = 50000          # Ignore massive non-defect objects
MIN_ASPECT_RATIO = 0.35          # Holes are somewhat round
MAX_ASPECT_RATIO = 2.8           # Not elongated

# --- FILTER THRESHOLDS ---
# NOTE: These now work on the INVERTED image (White hole -> Black hole)
DARKNESS_THRESHOLD = 90          # Pixels must be darker than this (0-255)
DARKNESS_RATIO = 0.4             # 40% of the box must be "dark"
TEXTURE_THRESHOLD = 30           # Std Dev of pixel intensity
CIRCULARITY_THRESHOLD = 0.40     # 1.0 is perfect circle
MIN_CONTOUR_POINTS = 8           # Ignore simple geometric noise

def is_valid_hole(bbox, frame_gray_inverted, debug=False):
    """
    Validates if the detection is a real hole using the INVERTED frame.
    On inverted frame: Hole = Dark, Background = Light.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Safety boundary checks
    h, w = frame_gray_inverted.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h: return False

    roi = frame_gray_inverted[y1:y2, x1:x2]
    if roi.size == 0: return False

    # CHECK 1: Darkness (Inverted: Hole should be dark)
    # Since we inverted, a bright hole (255) is now dark (0).
    dark_pixels = np.count_nonzero(roi < DARKNESS_THRESHOLD)
    dark_ratio = dark_pixels / roi.size
    
    if dark_ratio < DARKNESS_RATIO:
        if debug: print(f"   ❌ Darkness failed: {dark_ratio:.2f} < {DARKNESS_RATIO}")
        return False
    if debug: print(f"   ✓ Dark Ratio: {dark_ratio:.2f}")

    # CHECK 2: Texture (Holes are relatively flat/uniform compared to complex wrinkles)
    texture_var = np.std(roi)
    if texture_var > TEXTURE_THRESHOLD:
        if debug: print(f"   ❌ Texture too high: {texture_var:.1f}")
        # return False # Optional: Comment out if false negatives occur
    if debug: print(f"   ✓ Texture: {texture_var:.1f}")

    # CHECK 3: Circularity / Shape
    # Find contours within the ROI to see if it looks like a hole
    # We threshold first to isolate the 'dark' hole
    _, bin_roi = cv2.threshold(roi, DARKNESS_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < CIRCULARITY_THRESHOLD:
                if debug: print(f"   ❌ Shape irregular (Line/Fold): {circularity:.2f}")
                return False
            if debug: print(f"   ✓ Circularity: {circularity:.2f}")

    if debug: print(f"   ✅ HOLE CONFIRMED")
    return True

# --- MAIN SETUP ---
print("⏳ Loading model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Camera Ready")
print("⌨️  Press 'd' to toggle DEBUG mode")
print("⌨️  Press 'q' to QUIT")

debug_mode = False

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- STEP 1: PREPROCESSING (THE FIX) ---
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # INVERT: Makes White Holes -> Dark Holes
    # This aligns with your model's training on dark defects
    frame_inv = cv2.bitwise_not(frame_gray)
    
    # Create 3-channel input for YOLO from the INVERTED image
    input_for_model = cv2.cvtColor(frame_inv, cv2.COLOR_GRAY2BGR)

    # --- STEP 2: INFERENCE ---
    results = model(input_for_model, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # Visualization setup
    annotated = frame.copy() # Draw on original frame (not inverted) for human view
    defect_count = 0

    for r in results:
        for box in r.boxes:
            # Get Coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            area = w * h
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

            # --- STEP 3: PRE-FILTERING (Fast) ---
            if not (MIN_DEFECT_AREA <= area <= MAX_DEFECT_AREA):
                continue # Skip size mismatch
            
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue # Skip wrinkles/lines

            # --- STEP 4: DEEP VALIDATION (Strict) ---
            # Pass the INVERTED frame so checks find "dark" pixels
            if is_valid_hole([x1, y1, x2, y2], frame_inv, debug=debug_mode):
                defect_count += 1
                
                # Calculate Center for Stamper
                cx, cy = int(x1 + w/2), int(y1 + h/2)
                
                # DRAW (Green for Valid)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                
                # Label
                label = f"HOLE: {area:.0f}px"
                coord_text = f"X:{cx} Y:{cy}"
                cv2.putText(annotated, label, (int(x1), int(y1)-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, coord_text, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                print(f"🔴 DEFECT FOUND at {cx}, {cy} | Area: {area}")

    # --- DISPLAY ---
    if defect_count > 0:
        cv2.putText(annotated, f"!!! STOP MACHINE ({defect_count}) !!!", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(annotated, "INSPECTION ACTIVE - CLEAN", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show both views for debugging
    # Scale down inverted view and overlay it picture-in-picture
    inv_small = cv2.resize(frame_inv, (320, 180))
    annotated[0:180, 0:320] = cv2.cvtColor(inv_small, cv2.COLOR_GRAY2BGR)
    cv2.putText(annotated, "AI VIEW (Inverted)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imshow("Ultra-Strict Defect Detection", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('d'): 
        debug_mode = not debug_mode
        print(f"Debug Mode: {debug_mode}")

cap.release()
cv2.destroyAllWindows()