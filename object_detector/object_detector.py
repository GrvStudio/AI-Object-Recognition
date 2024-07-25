import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from utils.draw_border import draw_border
from utilsV2.barcode_detector import BarcodeDetector
from utilsV2.barcode_utils import BarcodeUtils

class ObjectDetector:
    def __init__(self, model_path, capture_interval=20):
        self.model = YOLO(model_path)
        self.cardboard_index = 0
        self.cardboard_indices = {}
        self.last_capture_time = 0
        self.capture_interval = capture_interval
        self.last_barcode_data = ""
        self.barcode_utils = BarcodeUtils()
        self.barcode_detector = BarcodeDetector()

    def detect_objects(self, frame):
        results = self.model.track(frame, persist=True, conf=0.5)
        return results

    def capture_cardboard_if_no_barcode_detected(self, frame):
        current_time = time.time()
        cardboard_detected = False
        barcode_detected = False
        id_card_detected = False
        capture_frame = False

        results = self.detect_objects(frame.copy())

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0].item())
                label = self.model.names[cls]
                confidence = box.conf[0].item()
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if label == "Cardboard":
                    self.cardboard_index += 1
                    cardboard_detected = True
                    border_color = (0, 0, 255)
                    frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)
                    if self.cardboard_index not in self.cardboard_indices:
                        self.cardboard_indices[self.cardboard_index] = {'bbox': (x1, y1, x2, y2), 'frame': frame.copy()}
                elif label == "ID Card":
                    id_card_detected = True
                    border_color = (255, 0, 0)
                    frame = draw_border(frame, (x1, y1), (x2, y2), color=border_color, thickness=2, line_length_x=30, line_length_y=30, padding=20)
                elif label == "Barcode":
                    barcode_detected = True
                    barcode_detector = cv2.barcode.BarcodeDetector()
                    overlay = frame.copy()
                    retval, points = barcode_detector.detect(overlay)
                    if retval:
                        points = points.astype(np.int64)
                        for point in points:
                            cv2.drawContours(overlay, [point], 0, (0, 255, 0), 2)
                        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        if len(self.cardboard_indices) > 0 and cardboard_detected and id_card_detected and not barcode_detected:
            capture_frame = True

        if current_time - self.last_capture_time >= self.capture_interval and capture_frame:
            os.makedirs('no-barcode', exist_ok=True)
            img_filename = f'no-barcode/captured_cardboard_{len(os.listdir("no-barcode")) + 1}.png'
            cv2.imwrite(img_filename, frame)
            print(f"Captured 'Cardboard' without barcode: {img_filename}")
            self.last_capture_time = current_time

    def process_frame(self, frame, cap):
        self.capture_cardboard_if_no_barcode_detected(frame)
        _, barcode_info = self.barcode_detector.detect_bar_code(frame.copy(), frame, cap)
        if barcode_info:
            filtered_barcodes = [info[0] for info in barcode_info if "PJA2406220" in info[0]]
            if filtered_barcodes:
                new_barcode_data = ' | '.join(filtered_barcodes)
                if new_barcode_data != self.last_barcode_data:
                    self.last_barcode_data = new_barcode_data
                    self.barcode_utils.play_sound()
                    self.barcode_utils.save_barcode_to_csv(self.last_barcode_data)
                    self.barcode_utils.update_barcode_count('detected_barcodes.csv')
        total_unique_barcodes = self.barcode_utils.get_total_unique_barcodes()
        self.barcode_utils.show_barcode_results(frame, self.last_barcode_data, total_unique_barcodes)
        return frame