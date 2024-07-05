
import csv
import os

# Fungsi untuk memuat barcode yang sudah ada dari file CSV saat aplikasi dimulai
# Set untuk melacak barcode yang sudah disimpan
saved_barcodes = set()
def load_saved_barcodes():
    file_path = 'detected_barcodes.csv'
    if os.path.isfile(file_path):
        with open(file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                saved_barcodes.add(row[0])

# Fungsi untuk menyimpan data barcode ke file CSV
def save_barcode_to_csv(barcode_data):
    file_path = 'detected_barcodes.csv'

    # Memuat barcode yang sudah ada jika set saved_barcodes masih kosong
    if not saved_barcodes:
        load_saved_barcodes()

    # Cek apakah barcode_data sudah disimpan sebelumnya
    if barcode_data not in saved_barcodes:
        with open(file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([barcode_data])
            saved_barcodes.add(barcode_data)