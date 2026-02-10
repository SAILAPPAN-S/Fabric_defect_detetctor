# Fabric Defect Detection System

An AI-powered textile defect detection system using YOLOv8 for real-time quality control in fabric manufacturing. This system enables automated inspection of textile products to identify defects with high accuracy, reducing manual inspection time and improving production quality.

## 🎯 Overview

This project implements a deep learning-based defect detection system specifically designed for textile manufacturing. It leverages the YOLOv8 architecture to detect fabric defects in real-time, providing multiple deployment options for various industrial scenarios.

## ✨ Features

- **Real-time Detection**: Live camera feed processing with immediate defect notification
- **API Integration**: RESTful API for integration with existing manufacturing systems
- **Industrial Ready**: Configurable confidence thresholds and PLC integration capabilities
- **Flexible Deployment**: Support for camera monitoring and API-based processing

## 🧠 Model Architecture

**Model**: YOLOv8 (You Only Look Once v8)
- **Framework**: Ultralytics YOLOv8
- **Task**: Object Detection
- **Classes**: 1 (Defect)
- **Input**: RGB Images or Monochrome Images
- **Architecture**: YOLOv8n (Nano variant for optimal speed-accuracy balance)

### Trained Models
- `best.pt` - Best performing model checkpoint (recommended for production)
- `last.pt` - Latest training checkpoint

## 📊 Dataset

**Dataset**: AITEX Textile Defects v2
- **Source**: Roboflow Universe
- **License**: CC BY 4.0
- **Classes**: 1 (Defect)
- **Splits**: 
  - Training Set
  - Validation Set
  - Test Set

The AITEX dataset contains images of various textile defects commonly found in fabric manufacturing processes.

## 📁 Project Structure

```
fabric_detection/
├── app.py                          # Flask API server for defect detection
├── main.py                         # Real-time camera detection system
├── Fabric_defect_detector.ipynb    # Training and experimentation notebook
├── best.pt                         # Best trained model (primary)
├── last.pt                         # Latest checkpoint
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore file
```

**Note**: The dataset and base YOLO models are not included in the repository due to size constraints. See the [Setup](#-installation) section for download instructions.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd fabric_detection/final
```

2. **Install dependencies**
```bash
pip install ultralytics opencv-python flask pillow
```

3. **Download the dataset** (Optional - only needed for training)

The AITEX Textile Defects dataset is available on Roboflow:
- Visit: https://universe.roboflow.com/sailappan-sn6vd/aitex-textile-defects-v2-9gyiw
- Download in YOLOv8 format
- Extract to `aitex-textile-defects-v2-1/` directory

4. **Verify model files**

The trained models (`best.pt` and `last.pt`) are included in the repository. If you need to retrain:
- Download base YOLOv8 model: `yolov8n.pt` will be auto-downloaded by Ultralytics
- Use the Jupyter notebook to train with your dataset

## 💻 Usage

### 1. Real-Time Camera Detection (`main.py`)

For live fabric inspection with a USB camera:

```bash
python main.py
```

**Configuration Options:**
- `MODEL_PATH`: Path to model file (default: `best.pt`)
- `CAMERA_INDEX`: Camera ID (default: `0`)
- `CONFIDENCE_THRESHOLD`: Detection sensitivity (default: `0.45`)
  - Lower values (0.30-0.45): Higher recall, more detections
  - Higher values (0.50-0.70): Higher precision, fewer false alarms

**Controls:**
- Press `q` to quit

**Features:**
- Real-time defect detection
- Visual alerts with red border on defect detection
- Console logging of detection events
- Simulated PLC signal interface for machine control

### 2. REST API Server (`app.py`)

For integration with web applications or manufacturing systems:

```bash
python app.py
```

**API Endpoint:**
```
POST /predict
Content-Type: multipart/form-data
Parameter: image (file)
```

**Example Request (curl):**
```bash
curl -X POST -F "image=@fabric_sample.jpg" http://localhost:5000/predict
```

**Example Response:**
```json
{
  "message": "Success",
  "defects_found": 2,
  "detections": [
    {
      "label": "Defect",
      "confidence": 0.87,
      "box": {
        "x1": 120.5,
        "y1": 340.2,
        "x2": 280.8,
        "y2": 450.6
      }
    }
  ]
}
```

**Server Configuration:**
- Host: `0.0.0.0` (accessible from network)
- Port: `5000`
- Confidence: `0.25` (adjustable in code)

## 🎯 Model Performance

**Recommended Thresholds:**
- **Production/Critical**: 0.50-0.60 (minimize false positives)
- **Quality Assurance**: 0.40-0.50 (balanced)
- **Initial Screening**: 0.25-0.40 (high recall)

**Optimization:**
- Use `best.pt` for highest accuracy
- Enable GPU (`device='cuda:0'`) for faster processing
- Adjust confidence based on defect severity requirements

## 🏭 Industrial Applications

- **Quality Control**: Automated inspection in textile production lines
- **Defect Classification**: Real-time identification of fabric anomalies
- **Process Optimization**: Data collection for manufacturing improvements
- **Cost Reduction**: Minimize manual inspection labor
- **Traceability**: Log defect occurrences for quality tracking

## 🔧 Integration with Industrial Systems

The system is designed for easy integration with:
- **PLC Systems**: Modbus TCP/RTU for machine control signals
- **SCADA**: API integration for centralized monitoring
- **MES**: Manufacturing Execution System data exchange
- **Quality Management Systems**: Defect logging and reporting

## 📝 Training Your Own Model

Use the included Jupyter notebook for training:

```bash
jupyter notebook Fabric_defect_detector.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model training configuration
- Hyperparameter tuning
- Evaluation metrics
- Export trained models

## ⚙️ Configuration Tips

### Camera Setup
```python
# Optimize for industrial cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
```

### Performance Optimization
- Use GPU acceleration when available
- Adjust image resolution based on defect size
- Implement batch processing for offline inspection
- Optimize confidence thresholds for your specific use case

## 📄 License
Dataset: CC BY 4.0 
