import cv2
import os
import time
from ultralytics import YOLO
from utils.draw_border import draw_border
from utils.detect_bar_code import detect_bar_code, show_barcode_results, update_barcode_count, get_total_unique_barcodes, count_png_files_in_no_barcode_folder
from utils.saved_to_Csv import load_saved_barcodes, save_barcode_to_csv

# Inisialisasi model YOLO yang sudah dilatih
model = YOLO('./runs/detect/train/weights/best.pt')  # Ganti dengan path ke model yang sudah dilatih

# Variabel global untuk menyimpan indeks unik objek "Cardboard"
cardboard_index = 0
cardboard_indices = {}  # Kamus untuk menyimpan indeks masing-masing objek "Cardboard" yang terdeteksi

# Variabel untuk melacak waktu terakhir gambar diambil
last_capture_time = 0
capture_interval = 10  # Interval waktu (detik) antara capture

# Fungsi untuk mendeteksi objek dalam frame menggunakan YOLO
def detect_objects(frame):
    results = model.track(frame, persist=True)
    return results

# Fungsi untuk menangkap gambar objek "Cardboard" jika tidak ada barcode yang terdeteksi
def capture_cardboard_if_no_barcode_detected(frame):
    global last_capture_time
    current_time = time.time()
    
    if current_time - last_capture_time < capture_interval:
        return  # Jika waktu kurang dari capture_interval, kembali tanpa melakukan capture
    
    results = detect_objects(frame.copy())  # Copy frame untuk diproses
    
    cardboard_detected = False
    id_card_detected = False
    barcode_detected = False
    cardboard_indices = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0].item())  # Konversi tensor ke integer
            label = model.names[cls]  # Ambil nama kelas dari model
            confidence = box.conf[0].item()  # Konversi tensor ke float
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if label == "Cardboard":
                global cardboard_index
                cardboard_indices[cardboard_index] = {'bbox': (x1, y1, x2, y2), 'frame': frame.copy()}
                cardboard_detected = True
                cardboard_index += 1
                border_color = (0, 0, 255)
                frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)
            elif label == "ID Card":
                id_card_detected = True
                border_color = (255, 0, 0)  # Biru
                frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)
            elif label == "Barcode":
                barcode_detected = True  # Tandai bahwa barcode terdeteksi

    # Jika tidak ada barcode yang terdeteksi dan kedua objek "Cardboard" dan "ID Card" terdeteksi
    if cardboard_detected and id_card_detected and not barcode_detected:
        for idx, data in cardboard_indices.items():
            x1, y1, x2, y2 = data['bbox']
            cardboard_frame = data['frame'][y1:y2, x1:x2]  # Potong frame ke area cardboard
            # Simpan frame sebagai gambar PNG di folder 'no-barcode'
            os.makedirs('no-barcode', exist_ok=True)
            img_filename = f'no-barcode/captured_cardboard_{len(os.listdir("no-barcode")) + 1}.png'
            cv2.imwrite(img_filename, cardboard_frame)
            print(f"Captured 'Cardboard' without barcode: {img_filename}")
            last_capture_time = current_time

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

        # Panggil fungsi untuk menangkap "Cardboard" jika tidak ada barcode yang terdeteksi
        capture_cardboard_if_no_barcode_detected(frame)

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

        # Tampilkan hasil deteksi barcode di atas webcam
        show_barcode_results(frame, last_barcode_data, total_unique_barcodes)

        # Tampilkan frame dengan hasil deteksi
        cv2.imshow('AI Prototype', frame)

        # Break loop jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()