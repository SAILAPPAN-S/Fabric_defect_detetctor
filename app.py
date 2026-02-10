from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.25  # Adjust this based on your testing

# --- LOAD MODEL (Once at startup) ---
print("⏳ Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

@app.route('/', methods=['GET'])
def index():
    return "Fabric Defect Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to detect defects in an uploaded image.
    Expects a multipart/form-data request with an 'image' file.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # 1. Read the image file into memory
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))

        # 2. Run Inference
        results = model(img, conf=CONFIDENCE_THRESHOLD)

        # 3. Process Results
        detections = []
        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get confidence and class name
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]

                detections.append({
                    "label": cls_name,
                    "confidence": round(conf, 2),
                    "box": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    }
                })

        # 4. Return JSON Response
        response = {
            "message": "Success",
            "defects_found": len(detections),
            "detections": detections
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)