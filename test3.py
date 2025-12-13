import cv2
import os
from deepface import DeepFace
import numpy as np
import csv
from datetime import date

# ---------------- CONFIG ----------------
KNOWN_IMAGES_DIR = "known_images"
ATTENDANCE_FILE = "attendance.csv"

THRESHOLD = 28        # strict threshold
CONFIRM_FRAMES = 5    # multi-frame confirmation
MARGIN = 3.0          # confidence gap

# --------------------------------------------------------------------
# STEP 1: Build face database (MEAN EMBEDDINGS PER PERSON)
# --------------------------------------------------------------------
def build_face_db():
    mean_embeddings = []
    identities = []

    for person_name in os.listdir(KNOWN_IMAGES_DIR):
        person_folder = os.path.join(KNOWN_IMAGES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            try:
                emb = DeepFace.represent(
                    img_path=img_path,
                    model_name="Facenet512",
                    enforce_detection=False
                )[0]["embedding"]

                person_embeddings.append(emb)

            except Exception:
                pass

        # Compute mean embedding for this person
        if len(person_embeddings) > 0:
            mean_embeddings.append(np.mean(person_embeddings, axis=0))
            identities.append(person_name)

    return mean_embeddings, identities


# --------------------------------------------------------------------
# STEP 2: Recognize face (STRICT + CONFIDENT)
# --------------------------------------------------------------------
def recognize_face(face_img, db_embeddings, db_identities):
    try:
        rep = DeepFace.represent(
            face_img,
            model_name="Facenet512",
            enforce_detection=False
        )[0]["embedding"]
    except:
        return "Unknown", 999

    distances = np.linalg.norm(
        np.array(db_embeddings) - np.array(rep),
        axis=1
    )

    best_idx = np.argmin(distances)
    best_dist = distances[best_idx]

    sorted_dist = np.sort(distances)
    second_best = sorted_dist[1] if len(sorted_dist) > 1 else 999

    # Reject if not confident
    if best_dist < THRESHOLD and (second_best - best_dist) > MARGIN:
        return db_identities[best_idx], best_dist

    return "Unknown", best_dist


# --------------------------------------------------------------------
# STEP 3: Mark attendance
# --------------------------------------------------------------------
def mark_attendance(name, marked_today):
    today = date.today().isoformat()

    if name in marked_today:
        return

    file_exists = os.path.isfile(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date"])
        writer.writerow([name, today])

    marked_today.add(name)
    print(f"[ATTENDANCE] Marked: {name}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    face_db, identities = build_face_db()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    marked_today = set()
    recognition_counter = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            name, dist = recognize_face(face_img, face_db, identities)

            if name != "Unknown":
                recognition_counter[name] = recognition_counter.get(name, 0) + 1

                if recognition_counter[name] >= CONFIRM_FRAMES:
                    mark_attendance(name, marked_today)

                label = f"{name}"
                color = (0, 255, 0)
            else:
                recognition_counter.clear()
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
