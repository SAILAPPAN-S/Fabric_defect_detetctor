from ultralytics import YOLO
model = YOLO('notebook/best.pt')
print("Classes:", model.names)
