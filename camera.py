import cv2
import dlib
import numpy as np
from datetime import datetime
from win10toast import ToastNotifier
from tkinter import *

globalBool = True;

n = ToastNotifier()

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
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
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

class window_design:
    def __init__(self):
        window = Tk()
        window.title("Yawn Cam")
        window.config(bg="darkgrey")
        window.resizable(False, False)
        label_Name = Label(window, text="YawnCam", bg="darkgrey", font=("Arial Bold", 30))
        button_Start = Button(window, text="Start", font=("Arial", 20), height=2, width=10, command=self.main)
        button_Stop = Button(window, text="Stop", font=("Arial", 20), height=2, width=10, command=self.stop)
        button_Stats = Button(window, text="Stats", font=("Arial", 20), height=2, width=10)
        label_Buffer1 = Label(window, bg="darkgrey")
        label_Buffer2 = Label(window, bg="darkgrey")
        label_Buffer3 = Label(window, bg="darkgrey")

        label_Name.grid(row=0, column=3)
        label_Buffer1.grid(row=2, column=0)
        button_Start.grid(row=2, column=1)
        label_Buffer2.grid(row=3, column=0)
        button_Stop.grid(row=2, column=3)
        button_Stats.grid(row=2, column=5)
        label_Buffer3.grid(row=2, column=6)

        window.mainloop()

    def main(self):
        globalBool = True
        cap = cv2.VideoCapture(0)
        yawns = 0
        yawn_status = False

        start = datetime
        end = 0
        while globalBool is True:
            ret, frame = cap.read()
            image_landmarks, lip_distance = mouth_open(frame)

            prev_yawn_status = yawn_status

            if lip_distance > 25:
                yawn_status = True

                cv2.putText(frame, "Subject is Yawning", (50, 450),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                output_text = " Yawn Count: " + str(yawns + 1)

                cv2.putText(frame, output_text, (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127), 2)

            else:
                yawn_status = False

            if prev_yawn_status == True and yawn_status == False:
                yawns += 1
                start = datetime.now()#.timestamp()
                if end != 0 and abs(end.minute - start.minute) < 1:
                    n.show_toast("You keep yawning!", "Coffee? :^)", duration=5)
                end = datetime.now()#.timestamp()

            #cv2.imshow('Live Landmarks', image_landmarks)
            cv2.imshow('Yawn Detection', frame)

            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break

        cap.release()
        cv2.destroyAllWindows()


    def stop(self):
        globalBool = False

window_design()
