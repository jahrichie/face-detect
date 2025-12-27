import cv2
import os
from datetime import datetime
import time

PERSON_NAME = "Rich"

def create_folder(name):
    dataset_folder = "dataset"
    os.makedirs(dataset_folder, exist_ok=True)
    person_folder = os.path.join(dataset_folder, name)
    os.makedirs(person_folder, exist_ok=True)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)

    # Open default camera (try 0, 1, 2 if needed)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    time.sleep(2)

    photo_count = 0
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {photo_count} photos saved.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)
