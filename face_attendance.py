import face_recognition
import cv2
import numpy as np

known_encodings = []
known_names = []

def add_face(image_path, name):
    img = face_recognition.load_image_file(image_path)
    enc = face_recognition.face_encodings(img)[0]

    known_encodings.append(enc)
    known_names.append(name)


def recognize(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    names = []

    for enc in encodings:
        matches = face_recognition.compare_faces(known_encodings, enc)
        name = "Unknown"

        if True in matches:
            idx = np.argmin(face_recognition.face_distance(known_encodings, enc))
            name = known_names[idx]

        names.append(name)

    return locations, names