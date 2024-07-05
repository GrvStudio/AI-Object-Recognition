import cv2
from ultralytics import YOLO
import csv
from utils.draw_border import draw_border
from utils.detect_bar_code import detect_bar_code, show_barcode_results, update_barcode_count, get_total_unique_barcodes
from utils.saved_to_Csv import load_saved_barcodes, save_barcode_to_csv

# Inisialisasi model YOLO yang sudah dilatih
model = YOLO('./runs/detect/train/weights/best.pt')  # Ganti dengan path ke model yang sudah dilatih

# Fungsi untuk mendeteksi objek dalam frame menggunakan YOLO
def detect_objects(frame):
    results = model.predict(source=frame)
    return results

# Buka webcam
cap = cv2.VideoCapture(0)  # 0 adalah indeks default untuk webcam internal

# Set pengaturan kamera untuk memastikan kecepatan frame tinggi
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

# Variabel untuk menyimpan nilai barcode terakhir yang terdeteksi
last_barcode_data = ""
# Memuat barcode yang sudah ada jika file detected_barcodes.csv sudah ada
load_saved_barcodes()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari webcam.")
            break

        # Deteksi QR code atau barcode
        _, barcode_info = detect_bar_code(frame.copy())  # Copy frame untuk diproses

        # Jika ada barcode yang terdeteksi, perbarui nilai terakhir jika berbeda
        if barcode_info:
            new_barcode_data = ' | '.join([info[0] for info in barcode_info])
            if new_barcode_data != last_barcode_data:
                last_barcode_data = new_barcode_data
                save_barcode_to_csv(last_barcode_data)  # Simpan data barcode ke CSV
                update_barcode_count(last_barcode_data)  # Perbarui jumlah deteksi barcode

        # Dapatkan jumlah total deteksi barcode yang berbeda
        total_unique_barcodes = get_total_unique_barcodes()

        # Deteksi objek dalam frame menggunakan YOLO
        results = detect_objects(frame.copy())  # Copy frame untuk diproses

        # Render hasil deteksi YOLO ke frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())  # Konversi tensor ke integer
                label = model.names[cls]  # Ambil nama kelas dari model
                confidence = box.conf[0].item()  # Konversi tensor ke float

                # Tambahkan teks label di atas bounding box YOLO
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Gambar border untuk objek non-barcode
                if label == "ID Card":
                    border_color = (255, 0, 0)  # Biru
                    frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)
                elif label == "Barcode":
                    # Jangan gambar border untuk Barcode
                    continue
                else:
                    border_color = (0, 255, 0)  # Hijau (default untuk label lainnya)
                    frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)

        # Tampilkan hasil deteksi barcode di atas webcam
        show_barcode_results(frame, last_barcode_data, total_unique_barcodes)

        # Tampilkan frame dengan hasil deteksi
        cv2.imshow('YOLOv8 Webcam Detection', frame)

        # Break loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()