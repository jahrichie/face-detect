import face_recognition
import cv2
import numpy as np
import time
import pickle
from gpiozero import LED

# LED setup
led = LED(17)  # change pin if needed

# Load encodings
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

time.sleep(2)

fps_start = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    names = []
    recognized = False

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, m in enumerate(matches) if m]
            counts = {}
            for i in matched_idxs:
                counts[known_face_names[i]] = counts.get(known_face_names[i], 0) + 1
            name = max(counts, key=counts.get)
            recognized = True

        names.append(name)

    # LED control
    if recognized:
        led.on()
    else:
        led.off()

    # Draw boxes
    for (top, right, bottom, left), name in zip(face_locations, names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # FPS
    frame_count += 1
    elapsed = time.time() - fps_start
    fps = frame_count / elapsed
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
led.off()
