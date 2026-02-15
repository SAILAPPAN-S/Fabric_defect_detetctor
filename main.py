import cv2
from ultralytics import YOLO
import numpy as np

# --- ULTRA-STRICT HOLE-ONLY DETECTION ---
MODEL_PATH = 'notebook/best.pt'
CONFIDENCE_THRESHOLD = 0.90      # VERY high - only strongest detections
MIN_DEFECT_AREA = 3000           # Larger minimum (ignore tiny noise)
MAX_DEFECT_AREA = 40000          # Upper limit (ignore backgrounds/humans)
MIN_ASPECT_RATIO = 0.35          # Holes are somewhat round
MAX_ASPECT_RATIO = 2.8           # Not elongated

# Hole-specific thresholds
DARKNESS_THRESHOLD = 90          # Holes are noticeably darker
DARKNESS_RATIO = 0.5             # 50% of hole must be dark
TEXTURE_THRESHOLD = 20           # Low texture (folds have high texture)
CIRCULARITY_THRESHOLD = 0.45     # Holes are circular (not linear folds)
EDGE_DENSITY_MIN = 0.02          # Holes have clear edges
EDGE_DENSITY_MAX = 0.25          # But not too many (not grain patterns)
MIN_CONTOUR_POINTS = 8           # Well-defined shape

def is_valid_hole(bbox, frame_gray, frame_bgr, debug=False):
    """Multi-layer hole validation - VERY strict"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Safety checks
    if x1 < 0 or y1 < 0 or x2 > frame_gray.shape[1] or y2 > frame_gray.shape[0]:
        return False
    
    roi_gray = frame_gray[y1:y2, x1:x2]
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    
    if roi_gray.size == 0:
        return False
    
    # CHECK 1: Darkness (holes expose darker fabric)
    dark_pixels = np.sum(roi_gray < DARKNESS_THRESHOLD)
    dark_ratio = dark_pixels / roi_gray.size
    if dark_ratio < DARKNESS_RATIO:
        if debug: print(f"   ❌ Darkness check failed: ratio={dark_ratio:.2f}")
        return False
    if debug: print(f"   ✓ Dark enough: {dark_ratio:.2f}")
    
    # CHECK 2: Low texture (folds have high variance)
    texture_var = np.std(roi_gray)
    if texture_var > TEXTURE_THRESHOLD:
        if debug: print(f"   ❌ Texture too high: {texture_var:.1f}")
        return False
    if debug: print(f"   ✓ Low texture: {texture_var:.1f}")
    
    # CHECK 3: Edges (holes have defined boundaries)
    edges = cv2.Canny(roi_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if not (EDGE_DENSITY_MIN <= edge_density <= EDGE_DENSITY_MAX):
        if debug: print(f"   ❌ Edge density wrong: {edge_density:.3f}")
        return False
    if debug: print(f"   ✓ Edge density: {edge_density:.3f}")
    
    # CHECK 4: Circularity (holes are round, folds are linear)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        if debug: print(f"   ❌ No contours found")
        return False
    
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < MIN_CONTOUR_POINTS:
        if debug: print(f"   ❌ Contour too simple: {len(cnt)} points")
        return False
    
    cont_area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * cont_area / (perimeter ** 2)
    if circularity < CIRCULARITY_THRESHOLD:
        if debug: print(f"   ❌ Not circular: {circularity:.2f}")
        return False
    if debug: print(f"   ✓ Circularity: {circularity:.2f}")
    
    # CHECK 5: Color analysis (holes might have slight color shift)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Check saturation is not too high (rules out colored patterns)
    saturation = hsv[:,:,1]
    avg_saturation = np.mean(saturation)
    if avg_saturation > 150:  # High saturation = likely pattern/person
        if debug: print(f"   ❌ Too saturated (pattern/person): {avg_saturation:.1f}")
        return False
    if debug: print(f"   ✓ Saturation OK: {avg_saturation:.1f}")
    
    # CHECK 6: Reject if too uniform (solid objects like people)
    brightness_variance = np.std(roi_gray)
    if brightness_variance < 5:  # Too uniform = solid object
        if debug: print(f"   ❌ Too uniform (solid object): {brightness_variance:.1f}")
        return False
    
    if debug: print(f"   ✅ ALL CHECKS PASSED - HOLE CONFIRMED")
    return True

def smart_hole_filter(model, frame, debug=False):
    """Detect ONLY real holes - VERY aggressive filtering"""
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # High confidence threshold
    results = model(frame_rgb, conf=CONFIDENCE_THRESHOLD, iou=0.7, verbose=False)
    
    valid_holes = []
    for r in results:
        if len(r.boxes) == 0:
            continue
            
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2-x1) * (y2-y1)
            aspect_ratio = min((x2-x1)/(y2-y1) + 0.001, (y2-y1)/(x2-x1) + 0.001)
            
            # Size and shape check first (fast)
            if not (MIN_DEFECT_AREA <= area <= MAX_DEFECT_AREA):
                continue
            if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                continue
            
            # Expensive checks only if size/shape OK
            if is_valid_hole([x1, y1, x2, y2], frame_gray, frame, debug=debug):
                valid_holes.append(box)
    
    return results, len(valid_holes) > 0, valid_holes

# --- LOAD MODEL ---
print("⏳ Loading model for HOLE-ONLY detection...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded - ULTRA-STRICT mode")
print("   Confidence: 0.90 | Detects: HOLES ONLY")
print("   Rejects: Folds, grains, humans, noise, lines")

# --- CAMERA ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✅ Camera ready | Press 'q'=quit | 'd'=debug mode")

# --- MAIN LOOP ---
frame_count = 0
holes_found = 0
debug_mode = False

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    
    # Detect only real holes
    results, hole_detected, valid_holes = smart_hole_filter(model, frame, debug=debug_mode)
    annotated = results[0].plot()
    
    # Draw valid holes in GREEN
    for i, hole in enumerate(valid_holes, 1):
        x1, y1, x2, y2 = hole.xyxy[0].cpu().numpy()
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(annotated, f"HOLE #{i}", (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Status display
    if hole_detected:
        cv2.rectangle(annotated, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
        cv2.putText(annotated, f"🕳️ HOLE CONFIRMED ({len(valid_holes)})", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
        holes_found += 1
        print(f"🟢 Frame {frame_count}: {len(valid_holes)} REAL HOLES detected")
    else:
        cv2.putText(annotated, "✅ CLEAN / NO HOLES", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    # Info bar
    cv2.putText(annotated, f"Conf: {CONFIDENCE_THRESHOLD} | Frame: {frame_count}", 
                (30, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow('HOLE-ONLY Detection (Ultra-Strict)', annotated)
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Complete: {holes_found} real holes in {frame_count} frames")