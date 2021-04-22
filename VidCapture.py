import os
import sys

import cv2
import keyboard
import numpy as np
import pyvirtualcam

video = cv2.VideoCapture(0)
width = int(video.get(3))
height = int(video.get(4))
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
haar_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
glasses = cv2.imread("assets/eye.png", -1)
run = True


def four_cams():
    image = np.zeros(frame.shape, np.uint8)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    image[:height // 2, :width // 2] = small_frame
    image[height // 2:, :width // 2] = small_frame
    image[:height // 2, width // 2:] = cv2.rotate(small_frame, cv2.ROTATE_180)
    image[height // 2:, width // 2:] = cv2.rotate(small_frame, cv2.ROTATE_180)
    return cv2.imshow('Frame', image)


def extract_blue():
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_blue = np.array([90, 50, 50])
    high_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, low_blue, high_blue)
    compared = cv2.bitwise_and(frame, frame, mask=mask)
    return cv2.imshow('Frame', compared)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def apply_invert():
    neg = cv2.bitwise_not(frame)
    return cv2.imshow('Frame', neg)


def verify_alpha_channel(frame):
    try:
        frame.shape[3]  # 4th position
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame


def apply_color_overlay(frame, intensity=0.2, blue=0, green=0, red=0):
    frame = verify_alpha_channel(frame)
    frame_h, frame_w, frame_c = frame.shape
    color_bgra = (blue, green, red, 1)
    overlay = np.full((frame_h, frame_w, 4), color_bgra, dtype='uint8')
    cv2.addWeighted(overlay, intensity, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def apply_sepia(frame, intensity=0.5):
    blue, green, red = 20, 66, 112
    frame = apply_color_overlay(frame, intensity=intensity, blue=blue, green=green, red=red)
    return cv2.imshow('Frame', frame)


def emoji_filter():
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        region_gray = gray[y:y + h, x:x + w]
        region_color = frame[y:y + h, x:x + w]

        eyes = haar_eye_cascade.detectMultiScale(region_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            glasses2 = image_resize(glasses.copy(), width=ew)

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    if glasses2[i, j][3] != 0:
                        region_color[ey + i, ex + j] = glasses2[i, j][0]

    return cv2.imshow('Frame', frame)


def face_recognition():
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        region_gray = gray[y:y + h, x:x + w]
        region_color = frame[y:y + h, x:x + w]

        eyes = haar_eye_cascade.detectMultiScale(region_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(region_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
    return cv2.imshow('Frame', frame)


def blur():
    kernel = np.ones((15, 15), np.float32) / 225
    smoothed = cv2.filter2D(frame, -1, kernel)
    return cv2.imshow('Blur', smoothed)


def erosion():
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.erode(frame, kernel, iterations=1)
    return cv2.imshow('Frame', image)


def edges():
    edges = cv2.Canny(frame, 80, 80)
    return cv2.imshow('Frame', edges)


def alpha_blend(frame_1, frame_2, mask):
    alpha = mask / 255.0
    blended = cv2.convertScaleAbs(frame_1 * (1 - alpha) + frame_2 * alpha)
    return blended


def apply_portrait_mode(frame):
    frame = verify_alpha_channel(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    blured = cv2.GaussianBlur(frame, (21, 21), 11)
    blended = alpha_blend(frame, blured, mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return cv2.imshow('Frame', frame)


print("\nType from q to p to add effect\nq -> Erosion\nw -> Four cams\ne -> Emoji"
      "\nr -> Extract blue\nt -> Invert\ny -> Sepia\nu -> Face detection\ni -> Blur\no -> Edges"
      "\np -> Portrait")
choice = input()
choice = str(choice)
print("\nPress F to restart")

with pyvirtualcam.Camera(width=1280, height=720, fps=30) as cam:
    print(f'Using virtual camera: {cam.device}')
    frames = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
    while run:
        ret, frame = video.read()
        frames[:, :, :3] = cam.frames_sent % 255  # grayscale animation
        cam.send(frames)
        cam.sleep_until_next_frame()
        # IDLEs might only close the app, works from shell
        if keyboard.is_pressed('f'):
            os.execv(sys.executable, [sys.executable] + sys.argv)

        dictionary = {
            'q': lambda: erosion(),
            'w': lambda: four_cams(),
            'e': lambda: emoji_filter(),
            'r': lambda: extract_blue(),
            't': lambda: apply_invert(),
            'y': lambda: apply_sepia(frame),
            'u': lambda: face_recognition(),
            'i': lambda: blur(),
            'o': lambda: edges(),
            'p': lambda: apply_portrait_mode(frame),
        }.get(choice)()
        if cv2.waitKey(1) == ord('q'):
            break

video.release()
cv2.destroyAllWindows()
