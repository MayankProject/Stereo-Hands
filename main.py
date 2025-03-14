import cv2
import mediapipe as mp
import numpy as np
import librosa
import pygame
import scipy.interpolate

file_path = "Track1.wav"
y, sr = librosa.load(file_path, sr=None)

pygame.mixer.init()

sound = pygame.mixer.Sound(file_path)
channel = sound.play()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 - 20

width, height = 800, 600  
waveform_height = height // 3  
chunk_size = sr // 30  
frame_index = 0


def downsample_waveform(waveform, target_points=100):
    x = np.linspace(0, len(waveform) - 1, num=len(waveform))  
    f = scipy.interpolate.interp1d(x, waveform, kind='cubic')  
    x_new = np.linspace(0, len(waveform) - 1, num=target_points)  
    return f(x_new)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  
    frame = cv2.resize(frame, (width, height))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        h, w, c = frame.shape
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            x_index, y_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)

            cv2.line(frame, (x_index, y_index), (x_thumb, y_thumb), (0, 255, 0), 1, cv2.LINE_AA)
            print(w, h)
            cv2.line(frame, (400, 0), (400, h), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (x_index, y_index), 10, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(frame, (x_thumb, y_thumb), 10, (255, 255, 255), 3, cv2.LINE_AA)

        if len(results.multi_hand_landmarks) == 2:
            landmarks = results.multi_hand_landmarks

            index_finger_tip_1 = landmarks[0].landmark[8]
            index_finger_tip_2 = landmarks[1].landmark[8]

            thumb_tip_1 = landmarks[0].landmark[4]
            thumb_tip_2 = landmarks[1].landmark[4]

            x_index_1, y_index_1 = int(index_finger_tip_1.x * w), int(index_finger_tip_1.y * h)
            x_index_2, y_index_2 = int(index_finger_tip_2.x * w), int(index_finger_tip_2.y * h)
            x_thumb_1, y_thumb_1 = int(thumb_tip_1.x * w), int(thumb_tip_1.y * h)
            x_thumb_2, y_thumb_2 = int(thumb_tip_2.x * w), int(thumb_tip_2.y * h)
            middle_point_1 = (x_index_1 + x_thumb_1) // 2, (y_index_1 + y_thumb_1) // 2
            middle_point_2 = (x_index_2 + x_thumb_2) // 2, (y_index_2 + y_thumb_2) // 2

            dist_1 = dist([x_index_1, y_index_1], [x_thumb_1, y_thumb_1])
            dist_2 = dist([x_index_2, y_index_2], [x_thumb_2, y_thumb_2])
            dist_middle = dist(middle_point_1, middle_point_2)
            left = 0
            right = 0
            if x_index_1 > w / 2:
                right = dist_1
                left = dist_2
                channel.set_volume(max(0, left/300), max(0, right/300))
            else:
                left = dist_1
                right = dist_2
                channel.set_volume(max(0, left/300), max(0, right/300))


            start = frame_index * chunk_size
            end = min(start + chunk_size, len(y))
            frame_index += 1
            waveform = y[start:end]

            if len(waveform) > 0:
                waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())  
                waveform = (waveform * waveform_height).astype(np.int32)  
                waveform = downsample_waveform(waveform, target_points=50)

                num_samples = len(waveform)
                cv2.line(frame, middle_point_1, middle_point_2, (255, 255, 255), 1, cv2.LINE_AA)
                y_adjust = 100

                for i in range(1, num_samples-1):
                    x1 = int(middle_point_1[0] + (i - 1) * (middle_point_2[0] - middle_point_1[0]) / num_samples)
                    x2 = int(middle_point_1[0] + i * (middle_point_2[0] - middle_point_1[0]) / num_samples)
                    y1 = int(middle_point_1[1] + (i - 1) * (middle_point_2[1] - middle_point_1[1]) / num_samples - waveform[i - 1]) + y_adjust 
                    y2 = int(middle_point_1[1] + i * (middle_point_2[1] - middle_point_1[1]) / num_samples - waveform[i]) + y_adjust 
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Something Cool Breh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
