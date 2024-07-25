import cv2
import numpy as np
from pyzbar.pyzbar import decode
from concurrent.futures import ThreadPoolExecutor

class BarcodeDetector:
    def __init__(self):
        self.last_detected_barcode = None

    def detect_bar_code(self, frame, frameReal, cap):
        angles = [0, 20, 30, 60, 70, 90, 110, 120, 130, 150, 170, 180, 200, 220, 250, 270, 300, 310, 330, 360]
        barcodes = decode(frame)
        barcode_info = []

        if not barcodes:
            # Jika tidak ada barcode yang terdeteksi, coba beberapa sudut rotasi
            with ThreadPoolExecutor() as executor:
                rotated_frames = list(executor.map(lambda angle: self.rotate_image(frame, angle), angles))
                results = list(executor.map(decode, rotated_frames))
                for decoded_barcodes in results:
                    if decoded_barcodes:
                        barcodes = decoded_barcodes
                        break

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
            if barcode_data != self.last_detected_barcode:
                self.last_detected_barcode = barcode_data

        return frame, barcode_info

    def detect_bar_code_polygon(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                overlay = frame.copy()
                alpha = 0.2  # Transparansi 70%

                cv2.polylines(overlay, [approx], True, (0, 255, 0), 2)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                break
        
        return frame

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        return rotated