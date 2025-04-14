# Attendance-System-Using-LBPH

This project uses OpenCV to detect and recognize faces for an attendance system using your webcam.

## Algorithm Used

- **Face Detection**: Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)
- **Face Recognition**: LBPH (Local Binary Patterns Histogram) algorithm using OpenCV's `cv2.face.LBPHFaceRecognizer_create()`

## Files

- `main.py` – Captures face images and saves them to the `data/` folder.
- `trainer.py` – Trains a face recognizer model (`trainer.xml`) using the saved face images.
- `face_recog.py` – Uses the webcam to recognize faces and logs attendance in a CSV file.

## Requirements

- Python
- OpenCV
- NumPy
- Pillow
- Pandas

Install all dependencies using:
pip install opencv-python numpy pillow pandas
