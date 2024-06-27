from ultralytics import YOLO

# Buat objek model
model = YOLO('yolov8n.pt')  # Bisa juga memilih versi lain seperti yolov8s.pt, yolov8m.pt, dll.

# Latih model
model.train(data='./data.yaml', epochs=50, imgsz=640)