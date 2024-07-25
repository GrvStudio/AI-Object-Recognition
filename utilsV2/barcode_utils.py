import cv2
import os
import csv
import pygame

class BarcodeUtils:
    def __init__(self):
        pygame.mixer.init()
        self.barcode_counts = {}

    def play_sound(self):
        pygame.mixer.music.load("/Users/elang/Documents/PROJECT/AI/NIKE/AI-Object-Recognition/assets/sound/sound-detect.mp3")
        pygame.mixer.music.play()

    def update_barcode_count(self, file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return
        
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                barcode_data = row[0]
                if barcode_data in self.barcode_counts:
                    self.barcode_counts[barcode_data] += 1
                else:
                    self.barcode_counts[barcode_data] = 1

    def get_total_unique_barcodes(self):
        return len(self.barcode_counts)

    def count_png_files_in_no_barcode_folder(self):
        folder_path = 'no-barcode'
        if not os.path.exists(folder_path):
            print(f"Folder '{folder_path}' tidak ditemukan.")
            return 0
        
        png_files = [file for file in os.listdir(folder_path) if file.endswith('.png')]
        png_file_count = len(png_files)
        return png_file_count

    def show_barcode_results(self, frame, barcode_data_combined, total_unique_barcodes):
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
        png_file_count = self.count_png_files_in_no_barcode_folder()
        png_count_text = f'No Barcode = {png_file_count}'
        png_count_text_size = cv2.getTextSize(png_count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        png_count_text_x = frame.shape[1] - png_count_text_size[0] - 10
        png_count_text_y = 30

        # Gambar background kuning untuk teks di pojok kanan atas
        cv2.rectangle(frame, (png_count_text_x - 5, png_count_text_y - png_count_text_size[1] - 5), (png_count_text_x + png_count_text_size[0] + 5, png_count_text_y + 5), (0, 255, 255), -1)

        # Tampilkan teks hitam di atas background kuning di pojok kanan atas
        cv2.putText(frame, png_count_text, (png_count_text_x, png_count_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def save_barcode_to_csv(self, barcode_data, file_path='detected_barcodes.csv'):
        with open(file_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([barcode_data])

    def load_saved_barcodes(self, file_path='detected_barcodes.csv'):
        if not os.path.exists(file_path):
            return []
        
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            saved_barcodes = [row[0] for row in reader]
        return saved_barcodes