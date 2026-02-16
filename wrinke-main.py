import cv2
from ultralytics import YOLO
import numpy as np

# --- AGGRESSIVE WRINKLE REJECTION CONFIG ---
MODEL_PATH = 'notebook/best.pt'
CONFIDENCE_THRESHOLD = 0.50      # Higher = fewer wrinkles
MIN_DEFECT_AREA = 2000          # Larger = ignore small wrinkles  
MIN_ASPECT_RATIO = 0.2          # Reject very thin detections (wrinkle-like)
TEXTURE_THRESHOLD = 15          # High texture variance = wrinkles

def is_wrinkle_like(bbox, frame_gray):
    """Texture analysis - wrinkles have high local variance"""
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame_gray[y1:y2, x1:x2]
    
    # High std dev = wrinkle texture
    texture_var = np.std(roi)
    return texture_var > TEXTURE_THRESHOLD

def smart_defect_filter(model, frame):
    """Minimal preprocessing + aggressive filtering"""
    
    # 1. NO CLAHE - just grayscale for texture analysis
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. RGB for YOLO (your model expects 3 channels)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 3. Conservative inference
    results = model(frame_rgb, conf=CONFIDENCE_THRESHOLD, iou=0.65, verbose=False)
    
    # 4. MULTI-LAYER WRINKLE FILTER
    valid_defects = []
    for r in results:
        if len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2-x1) * (y2-y1)
                aspect_ratio = min((x2-x1)/(y2-y1), (y2-y1)/(x2-x1))
                
                # REJECT if:
                if (area > MIN_DEFECT_AREA and           # Large enough
                    aspect_ratio > MIN_ASPECT_RATIO and  # Not too thin
                    not is_wrinkle_like([x1,y1,x2,y2], frame_gray)):  # Not wrinkly texture
                    valid_defects.append(box)
    
    return results, len(valid_defects) > 0, valid_defects

# --- LOAD MODEL ---
print("Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Anti-wrinkle mode ACTIVE")

# --- CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("✅ Live! 'q'=quit, 'u'=less strict, 's'=more strict")

# --- MAIN LOOP ---
frame_count = 0
strict_mode = 0  # 0=normal, 1=strict, -1=loose

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # Smart detection
    results, defect_detected, valid_dets = smart_defect_filter(model, frame)
    annotated = results[0].plot()
    
    # Status display
    if defect_detected:
        cv2.rectangle(annotated, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 12)
        cv2.putText(annotated, "🚨 REAL DEFECT - STOP!", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        print(f"🔴 DEFECT #{frame_count}: {len(valid_dets)} valid boxes")
    else:
        cv2.putText(annotated, "✅ CLEAN FABRIC", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    
    # Mode info
    mode_text = {0:"NORMAL", 1:"STRICT", -1:"LOOSE"}[strict_mode]
    cv2.putText(annotated, f"Mode: {mode_text} Conf:{CONFIDENCE_THRESHOLD}", 
                (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow('Smart Fabric Inspection', annotated)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'):  # More strict
        CONFIDENCE_THRESHOLD = min(0.9, CONFIDENCE_THRESHOLD + 0.05)
        MIN_DEFECT_AREA = int(MIN_DEFECT_AREA * 1.2)
        strict_mode += 1
        print(f"Strict mode ↑ Conf={CONFIDENCE_THRESHOLD:.2f} Area={MIN_DEFECT_AREA}")
    elif key == ord('u'):  # Less strict  
        CONFIDENCE_THRESHOLD = max(0.6, CONFIDENCE_THRESHOLD - 0.05)
        MIN_DEFECT_AREA = int(MIN_DEFECT_AREA * 0.8)
        strict_mode -= 1
        print(f"Loose mode ↓ Conf={CONFIDENCE_THRESHOLD:.2f} Area={MIN_DEFECT_AREA}")

cap.release()
cv2.destroyAllWindows()
