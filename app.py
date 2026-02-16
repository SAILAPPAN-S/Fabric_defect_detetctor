from flask import Flask, jsonify, request
from ultralytics import YOLO
import cv2
import threading
import requests
import time

app = Flask(__name__)

MODEL_PATH = "notebook/best.pt"
CONF_THRESHOLD = 0.25
NODE_SERVER = "http://localhost:3000"

print("⏳ Loading model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded")

def is_new_defect(box_center, defect_id):
    """Check if this is a new defect or same defect moved slightly"""
    global tracked_defects
    
    cx, cy = box_center
    
    # Check if defect is close to any existing tracked defect
    for tracked_id, (last_x, last_y) in list(tracked_defects.items()):
        distance = ((cx - last_x) ** 2 + (cy - last_y) ** 2) ** 0.5
        
        # If close to an existing defect, it's the same one
        if distance < DEFECT_DISTANCE_THRESHOLD:
            # Update position
            tracked_defects[tracked_id] = (cx, cy)
            return False, tracked_id
    
    # No nearby tracked defect - this is NEW
    new_id = max(tracked_defects.keys(), default=0) + 1
    tracked_defects[new_id] = (cx, cy)
    return True, new_id

# State variables
camera_running = False
last_defect_time = 0
total_defects_found = 0
frames_processed = 0
tracked_defects = {}  # Track defect locations to avoid re-reporting
process_active = False  # Only count defects when process is running
last_process_active = False  # Track state changes
DEFECT_COOLDOWN = 0.5  # Minimum 0.5s between defect notifications
DEFECT_DISTANCE_THRESHOLD = 50  # pixels - if defect moves >50px, it's a new defect

def check_process_status():
    """Query Node server to check if process is currently active"""
    global process_active
    try:
        response = requests.get(f"{NODE_SERVER}/plc/process-status", timeout=3)
        data = response.json()
        process_active = data.get("processActive", False)
        return data
    except Exception as e:
        print(f"⚠️ Failed to get process status: {e}")
        return {"processActive": False}

def continuous_monitoring():
    """Continuously monitor camera and detect defects in real-time"""
    global camera_running, last_defect_time, total_defects_found, frames_processed, tracked_defects, process_active, last_process_active
    
    cap = cv2.VideoCapture(0)
    
    # Optimize camera settings for industrial use
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Camera not accessible")
        return
    
    print("✅ Camera connected - Continuous monitoring started")
    print("🎥 Display window opened (Press 'q' to quit)")
    print("⏳ Waiting for process to start...")
    camera_running = True
    status_check_counter = 0
    
    try:
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                continue
            
            frames_processed += 1
            
            # Check process status every 30 frames to avoid too many requests
            status_check_counter += 1
            if status_check_counter >= 30:
                status_check_counter = 0
                status_data = check_process_status()
                
                # Detect process state change
                if process_active and not last_process_active:
                    print(f"🟢 PROCESS STARTED - Tracking enabled (PID: {status_data.get('processId')})")
                    tracked_defects.clear()  # Reset tracking
                    total_defects_found = 0  # Reset counter
                elif not process_active and last_process_active:
                    print(f"🔴 PROCESS ENDED - Tracking disabled")
                    tracked_defects.clear()  # Clear tracking
                
                last_process_active = process_active
            
            # Run inference on current frame
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            defect_count = 0 if boxes is None else len(boxes)
            
            # Track new defects detected in this frame
            new_defects_in_frame = []
            
            # Draw detection boxes on frame
            if defect_count > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0].item()
                    
                    # Calculate center of bounding box
                    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Check if this is a new defect (only if process is active)
                    if process_active:
                        is_new, defect_id = is_new_defect(box_center, len(tracked_defects))
                    else:
                        is_new = False
                        defect_id = 0  # Don't track if process not active
                    
                    # Draw red rectangle for new defects, orange for tracked
                    color = (0, 0, 255) if is_new else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw confidence and defect ID
                    label = f"NEW #{defect_id}" if is_new and process_active else f"Idle #{defect_id}" if not process_active else f"Track #{defect_id}"
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if is_new and process_active:
                        new_defects_in_frame.append((defect_id, defect_count, confidence))
            
            # Display frame count, defect info, and statistics
            process_status_text = "PROCESS ACTIVE ✅" if process_active else "IDLE ⏳"
            status_color = (0, 255, 0) if process_active else (255, 255, 0)
            
            cv2.putText(frame, f"Defects in Frame: {defect_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Unique Defects: {total_defects_found}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0) if process_active else (100, 100, 100), 2)
            cv2.putText(frame, f"Currently Tracking: {len(tracked_defects)}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            cv2.putText(frame, f"Frames: {frames_processed}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, process_status_text, (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show the frame
            cv2.imshow("Fabric Defect Detection - Continuous Monitoring", frame)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ User pressed 'q' - Stopping...")
                camera_running = False
                break
            
            # POST to Node ONLY on NEW defects AND when process is active
            if new_defects_in_frame and process_active:
                current_time = time.time()
                if current_time - last_defect_time >= DEFECT_COOLDOWN:
                    last_defect_time = current_time
                    total_defects_found += len(new_defects_in_frame)
                    
                    try:
                        # Send each new defect
                        for defect_id, count, conf in new_defects_in_frame:
                            response = requests.post(
                                f"{NODE_SERVER}/plc/detect-result",
                                json={
                                    "defect": True,
                                    "count": count,
                                    "defectId": defect_id,
                                    "confidence": float(conf),
                                    "timestamp": current_time
                                },
                                timeout=5
                            )
                            print(f"📤 NEW Defect #{defect_id} POST sent to Node (confidence: {conf:.2f}, Total: {total_defects_found})")
                    except Exception as e:
                        print(f"⚠️ Failed to POST to Node: {e}")
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        camera_running = False
        print(f"🛑 Camera released - Session Stats: Total Unique Defects: {total_defects_found}, Frames Processed: {frames_processed}")

@app.route("/", methods=["GET"])
def home():
    return "AI Service - Continuous Monitoring"

@app.route("/status", methods=["GET"])
def status():
    """Check if camera is currently monitoring"""
    return jsonify({
        "camera_running": camera_running,
        "model_loaded": True
    })

@app.route("/detect", methods=["GET"])
def detect():
    """Legacy endpoint for backward compatibility (optional)"""
    if not camera_running:
        return jsonify({
            "defect": False,
            "count": 0,
            "error": "Continuous monitoring not active"
        }), 503
    
    return jsonify({
        "defect": False,
        "count": 0,
        "message": "Use continuous monitoring via POST"
    })

if __name__ == "__main__":
    # Start continuous monitoring in background thread
    monitor_thread = threading.Thread(target=continuous_monitoring, daemon=True)
    monitor_thread.start()
    print("🔄 Monitoring thread started")
    
    # Run Flask server
    app.run(host="0.0.0.0", port=5000)
