import cv2
import dlib
import numpy as np
import sqlite3
from datetime import datetime
import time
from win10toast import ToastNotifier

# Notifications
n = ToastNotifier()

# Database
conn = sqlite3.connect('data.db')
cur = conn.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS data (Event VARCHAR, Time VARCHAR)')
conn.commit()

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0

    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance


cap = cv2.VideoCapture(0)
yawn_status = False

start = datetime
end = 0
while True:
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)

    prev_yawn_status = yawn_status

    if lip_distance > 25:
        yawn_status = True

    else:
        yawn_status = False

    if prev_yawn_status == True and yawn_status == False:
        start = datetime.now()
        if end != 0 and abs(end.minute - start.minute) < 1:
            ts = time.time()
            timestamp = datetime.now()
            cur.execute("INSERT INTO data VALUES ('Drowsy', CURRENT_TIMESTAMP)")
            conn.commit()
            n.show_toast("You keep yawning!", "Coffee? :^)", duration=5)
        end = datetime.now()

    # cv2.imshow('Yawn Detection', frame)

    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

conn.close()
cap.release()
cv2.destroyAllWindows()
