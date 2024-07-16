import cv2
import os
import pygame
from pyzbar.pyzbar import decode

# Inisialisasi pygame untuk suara
pygame.mixer.init()

def play_sound():
    pygame.mixer.music.load("/Users/elang/Documents/PROJECT/AI/NIKE/AI-Object-Recognition/assets/sound/sound-detect.mp3")
    pygame.mixer.music.play()

# Variabel untuk melacak apakah suara telah diputar untuk barcode yang terdeteksi
last_detected_barcode = None

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

        # Memainkan suara jika barcode baru terdeteksi
        global last_detected_barcode
        if barcode_data != last_detected_barcode:
            play_sound()
            last_detected_barcode = barcode_data

        # Update jumlah deteksi barcode
        update_barcode_count(barcode_data)

    return frame, barcode_info

# Fungsi untuk menghitung jumlah file PNG dalam folder no-barcode
def count_png_files_in_no_barcode_folder():
    folder_path = 'no-barcode'
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' tidak ditemukan.")
        return 0
    
    png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
    png_file_count = len(png_files)
    return png_file_count

# Fungsi untuk menampilkan hasil deteksi barcode di atas webcam
def show_barcode_results(frame, barcode_data_combined, total_unique_barcodes):
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

    # Tampilkan jumlah total deteksi barcode di pojok kiri atas frame dengan background kuning dan teks hitam
    count_text = f'Count detect = {total_unique_barcodes}'
    count_text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    count_text_x = 10
    count_text_y = 30

    # Gambar background kuning untuk teks
    cv2.rectangle(frame, (count_text_x - 5, count_text_y - count_text_size[1] - 5), (count_text_x + count_text_size[0] + 5, count_text_y + 5), (0, 255, 255), -1)

    # Tampilkan teks hitam di atas background kuning
    cv2.putText(frame, count_text, (count_text_x, count_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Hitung jumlah file PNG dalam folder no-barcode
    png_file_count = count_png_files_in_no_barcode_folder()
    png_count_text = f'No Barcode = {png_file_count}'
    png_count_text_size = cv2.getTextSize(png_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    png_count_text_x = frame.shape[1] - png_count_text_size[0] - 10
    png_count_text_y = 30

    # Gambar background kuning untuk teks di pojok kanan atas
    cv2.rectangle(frame, (png_count_text_x - 5, png_count_text_y - png_count_text_size[1] - 5), (png_count_text_x + png_count_text_size[0] + 5, png_count_text_y + 5), (0, 255, 255), -1)

    # Tampilkan teks hitam di atas background kuning di pojok kanan atas
    cv2.putText(frame, png_count_text, (png_count_text_x, png_count_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Fungsi untuk memperbarui jumlah deteksi barcode
barcode_counts = {}
def update_barcode_count(barcode_data):
    if barcode_data in barcode_counts:
        barcode_counts[barcode_data] += 1
    else:
        barcode_counts[barcode_data] = 1

# Fungsi untuk mendapatkan jumlah total deteksi barcode yang berbeda
def get_total_unique_barcodes():
    return len(barcode_counts)