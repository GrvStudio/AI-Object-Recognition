import cv2
from object_detector.object_detector import ObjectDetector

def main():
    detector = ObjectDetector(model_path='./runs/detect/train/weights/best.pt')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Contoh kontrol fokus otomatis, ganti dengan API kamera yang sesuai
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Aktifkan auto focus

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari webcam.")
                break

            frame = detector.process_frame(frame, cap)
            cv2.imshow('AI Prototype', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()