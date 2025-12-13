
import cv2
import os
from deepface import DeepFace
import numpy as np
import csv
from datetime import date

# Path to folder containing known persons' folders
KNOWN_IMAGES_DIR = "known_images"
ATTENDANCE_FILE = "attendance.csv"

# --------------------------------------------------------------------
# STEP 1: Build a database of known faces
# --------------------------------------------------------------------
def build_face_db():
    embeddings = []
    identities = []

    for person_name in os.listdir(KNOWN_IMAGES_DIR):
        person_folder = os.path.join(KNOWN_IMAGES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            try:
                print(f"[INFO] Embedding: {img_path}")
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512"
                )[0]["embedding"]

                embeddings.append(embedding)
                identities.append(person_name)

            except Exception as e:
                print(f"[ERROR] Failed to embed {img_path}: {e}")

    print("[INFO] Embedding completed.")
    return embeddings, identities


# --------------------------------------------------------------------
# STEP 2: Recognize face from image crop
# --------------------------------------------------------------------
def recognize_face(face_img, db_embeddings, db_identities, threshold=40):
    try:
        rep = DeepFace.represent(
            face_img,
            model_name="Facenet512"
        )[0]["embedding"]
    except:
        return "Unknown"

    min_dist = 999
    identity = "Unknown"

    for idx, db_emb in enumerate(db_embeddings):
        dist = np.linalg.norm(np.array(rep) - np.array(db_emb))
        if dist < min_dist:
            min_dist = dist
            identity = db_identities[idx]

    if min_dist < threshold:
        return identity
    else:
        return "Unknown"


# --------------------------------------------------------------------
# STEP 3: Mark attendance in CSV
# --------------------------------------------------------------------
def mark_attendance(name, marked_today):
    today = date.today().isoformat()

    if name in marked_today:
        return

    file_exists = os.path.isfile(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            writer.writerow(["Name", "Date"])

        writer.writerow([name, today])

    marked_today.add(name)
    print(f"[ATTENDANCE] Marked present: {name} ({today})")


# --------------------------------------------------------------------
# MAIN FUNCTION — Webcam Recognition with Bounding Boxes
# --------------------------------------------------------------------
def main():
    print("[INFO] Building face database...")
    face_db, identities = build_face_db()

    print("[INFO] Loading face detector...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam not detected!")
        return

    marked_today = set()   # <-- Keeps track of attendance for today

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            name = recognize_face(face_img, face_db, identities)

            if name != "Unknown":
                mark_attendance(name, marked_today)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
