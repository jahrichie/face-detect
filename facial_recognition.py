import face_recognition
import cv2
import numpy as np
import time
import pickle

# =========================
# Configuration
# =========================

CAMERA_INDEX = 0          # Try 0, 1, 2 if needed
SCALE = 1.0               # 1.0 = NO scaling (full resolution)
FRAME_WIDTH = 1280        # Increase for better distance detection
FRAME_HEIGHT = 720

# =========================
# Load face encodings
# =========================

print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# =========================
# Initialize camera
# =========================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open camera")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

time.sleep(2)

# =========================
# Main loop
# =========================

fps_start = time.time()
frame_count = 0

print("[INFO] Starting face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame")
        break

    # Optional scaling (disabled when SCALE = 1.0)
    if SCALE != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
    else:
        small_frame = frame

    # Convert BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings,
            encoding,
            tolerance=0.5
        )

        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, m in enumerate(matches) if m]
            counts = {}

            for i in matched_idxs:
                counts[known_face_names[i]] = counts.get(known_face_names[i], 0) + 1

            name = max(counts, key=counts.get)

        face_names.append(name)

    # Draw results
    inv_scale = int(1 / SCALE)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= inv_scale
        right *= inv_scale
        bottom *= inv_scale
        left *= inv_scale

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # FPS counter
    frame_count += 1
    elapsed = time.time() - fps_start
    fps = frame_count / elapsed if elapsed > 0 else 0

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =========================
# Cleanup
# =========================

cap.release()
cv2.destroyAllWindows()
print("[INFO] Exiting cleanly.")
