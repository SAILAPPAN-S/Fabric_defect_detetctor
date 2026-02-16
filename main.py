import cv2
from ultralytics import YOLO
import numpy as np

# 🏭 FABRIC FOLDING MACHINE - HOLE & STAIN SPECIALIST

MODEL_PATH = 'notebook/best.pt'
CAMERA_INDEX = 0

# --- CLASS MAPPING (Check your data.yaml to confirm these IDs!) ---
CLASS_HOLE = 0
CLASS_STAIN = 1

# --- TUNING ---
CONFIDENCE_THRESHOLD = 0.40      
IGNORE_BORDER_PIXELS = 50        # Ignore edge noise

# --- HOLE SETTINGS (High Contrast) ---
HOLE_IS_BRIGHT = False           # False = Dark Hole (Top Light), True = White Hole (Backlight)
MIN_HOLE_AREA = 1000             # Holes can be small

# --- STAIN SETTINGS (Low Contrast - The Tricky Part) ---
MIN_STAIN_AREA = 3000            # Stains usually need to be bigger to be visible
MAX_STAIN_ASPECT_RATIO = 3.0     # Rejects Wrinkles! (Stains are round-ish, Wrinkles are long lines)

def is_wrinkle(w, h):
    """
    Stains are blobs. Wrinkles are lines.
    Returns True if the shape is too long/skinny (likely a wrinkle).
    """
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    if aspect_ratio > MAX_STAIN_ASPECT_RATIO:
        return True # It's a line (Wrinkle)
    return False # It's a blob (Stain/Hole)

def main():
    print("⏳ Loading AI Model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded! Tracking ONLY Holes & Stains.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Grayscale & Lighting Fix for HOLES
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if HOLE_IS_BRIGHT:
            # If looking for White Holes, Invert for the AI
            analysis_frame = cv2.bitwise_not(gray_frame)
        else:
            analysis_frame = gray_frame.copy()

        # 2. Inference (Filter for Holes & Stains ONLY)
        model_input = cv2.cvtColor(analysis_frame, cv2.COLOR_GRAY2BGR)
        
        # classes=[0, 1] tells YOLO to IGNORE everything else (threads, objects, etc.)
        results = model(model_input, conf=CONFIDENCE_THRESHOLD, classes=[CLASS_HOLE, CLASS_STAIN], verbose=False)
        
        defect_found = False
        display_frame = frame.copy()

        for r in results:
            for box in r.boxes:
                # Extract Data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                w, h = x2 - x1, y2 - y1
                area = w * h

                # --- GLOBAL FILTER: Border Noise ---
                if x1 < IGNORE_BORDER_PIXELS or y1 < IGNORE_BORDER_PIXELS or \
                   x2 > 1280 - IGNORE_BORDER_PIXELS or y2 > 720 - IGNORE_BORDER_PIXELS:
                    continue

                # DUAL LOGIC: Treat Holes and Stains Differently
        
                # --- CASE 1: HOLE LOGIC ---
                if cls_id == CLASS_HOLE:
                    if area < MIN_HOLE_AREA: continue # Ignore tiny pinholes if needed
                    
                    # Holes are usually High Contrast.
                    # Current model + lighting fix usually handles this well.
                    label = "HOLE"
                    color = (0, 0, 255) # Red

                # --- CASE 2: STAIN LOGIC ---
                elif cls_id == CLASS_STAIN:
                    # Stains are tricky. They look like wrinkles.
                    
                    # FILTER 1: Area (Stains are usually bigger than pinholes)
                    if area < MIN_STAIN_AREA: continue 

                    # FILTER 2: Shape (The Wrinkle Killer)
                    if is_wrinkle(w, h):
                        # It's long and skinny -> Likely a shadow/fold -> IGNORE
                        continue 
                    
                    label = "STAIN"
                    color = (0, 255, 255) # Yellow/Orange

                # If we survived the filters, it's a REAL defect
                defect_found = True
                
                # Draw Box
                cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(display_frame, f"{label} {area:.0f}px", (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                print(f"🔴 {label} FOUND at {int(x1)},{int(y1)}")

        # Status Display
        if defect_found:
            cv2.putText(display_frame, "!!! STOP MACHINE !!!", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            cv2.putText(display_frame, "CLEAN", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Hole & Stain Detector", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()