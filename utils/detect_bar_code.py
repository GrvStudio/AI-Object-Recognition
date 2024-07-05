import cv2
from pyzbar.pyzbar import decode


# Fungsi untuk mendeteksi dan membaca barcode dari frame
def detect_bar_code(frame):
    barcodes = decode(frame)
    barcode_info = []

    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')  # Konversi data barcode ke string
        barcode_type = barcode.type
        x, y, w, h = barcode.rect
        barcode_info.append((barcode_data, x, y, w, h))

        # Tentukan ukuran teks
        text_size = cv2.getTextSize(barcode_data, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_w, text_h = text_size

        # Tentukan posisi teks di atas atau di luar barcode
        text_x = x + (w - text_w) // 2
        text_y = y - 10  # Atur posisi teks 10 pixel di atas barcode

        # Pastikan teks tetap di dalam frame jika di bagian atas
        if text_y < 0:
            text_y = y + h + text_h  # Geser ke bawah barcode jika teks di luar frame atas

        # Gambar background putih untuk teks
        cv2.rectangle(frame, (text_x, text_y - text_h), (text_x + text_w, text_y + 10), (255, 255, 255), -1)

        # Tampilkan data barcode dengan teks merah di atas background putih
        cv2.putText(frame, f'{barcode_data}', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Cetak data barcode di konsol
        print(f'Detected {barcode_type} barcode: {barcode_data}')

    return frame, barcode_info


# Fungsi untuk menampilkan hasil deteksi barcode di atas webcam
def show_barcode_results(frame, barcode_data_combined):
    if barcode_data_combined:
        # Tentukan ukuran teks
        text_size = cv2.getTextSize(barcode_data_combined, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_w, text_h = text_size

        # Tentukan posisi teks di tengah atas frame
        text_x = (frame.shape[1] - text_w) // 2
        text_y = text_h + 10

        # Gambar background putih untuk teks
        cv2.rectangle(frame, (text_x - 10, text_y - text_h - 10), (text_x + text_w + 10, text_y + 10), (255, 255, 255), -1)

        # Tampilkan data barcode dengan teks merah di atas background putih
        cv2.putText(frame, barcode_data_combined, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
