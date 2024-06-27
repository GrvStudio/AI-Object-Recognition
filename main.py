import cv2
import os
import hashlib
# from ultralytics import YOLO
from utils.draw_border import draw_border

# Inisialisasi model YOLO dengan model yang sudah dilatih
model = YOLO('./runs/detect/train/weights/best.pt')

# Fungsi untuk mendeteksi objek dalam frame
def detect_objects(frame):
    results = model.predict(source=frame)
    return results

# Buka webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

frame_nmr = -1

try:
    while True:
        frame_nmr += 1
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari webcam.")
            break

        # Mengurangi noise dengan Median blur
        frame_blurred = cv2.medianBlur(frame, 5)

        results = detect_objects(frame_blurred)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())  # Konversi tensor ke integer
                label = model.names[cls]  # Ambil nama kelas dari model
                confidence = box.conf[0].item()  # Konversi tensor ke float

                # Cetak hasil deteksi
                print(f'Label: {label}, Confidence: {confidence:.2f}, BBox: ({x1}, {y1}), ({x2}, {y2})')

                # if label == 'Cardboard':
                draw_border(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=10, line_length_x=30, line_length_y=30)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tampilkan frame
        cv2.imshow('YOLOv8 Webcam Detection', frame)
        
        # Break loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()