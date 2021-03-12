import cv2
import numpy as np
CAPTURE_FRAMES = 100
ERODE_SIZE = 2
cam = cv2.VideoCapture(0)
img_counter = 0
FRAMES = []
counter = 0
backgroundModel = cv2.createBackgroundSubtractorMOG2()
while counter < CAPTURE_FRAMES:
    counter += 1
    ret, frame = cam.read()
    FRAMES.append(frame)
    backgroundModel.apply(frame)



for frame in FRAMES:
    counter += 1
    # ret, frame = cam.read()
    # if not ret:
    #     print("failed to grab frame")
    #     break

    hand_mask = backgroundModel.apply(frame, learningRate=0)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_SIZE,ERODE_SIZE))
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_ERODE, kern)

    cv2.imshow("Live", frame)
    cv2.imshow("Mask", hand_mask)

    k = cv2.waitKey(42)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1