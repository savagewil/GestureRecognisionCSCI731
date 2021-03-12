import cv2
import numpy as np
CAPTURE_FRAMES = 100
ERODE_SIZE = 2
cam = cv2.VideoCapture(0)
img_counter = 0
FRAMES = []
counter = 0

# while counter < CAPTURE_FRAMES:
#     counter += 1
#     ret, frame = cam.read()
#     FRAMES.append(frame)



while True:
    counter += 1
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, hand_mask) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(hand_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_masked = hand_mask.copy()
    # print(contours)
    for contour in contours:
        if 100 <= cv2.contourArea(contour):
            cv2.drawContours(contour_masked, [contour], 0, (127, 127, 127), 3)
            hull_shape = cv2.convexHull(contour, returnPoints=True)
            # print(hull_shape)
            hull = cv2.convexHull(contour, returnPoints=False)
            # print(hull)
            cv2.drawContours(contour_masked, [hull_shape], 0, (100, 100, 100), 3)
            try:
                # defects = cv2.convexityDefects(contour, hull)
                for point in contour:
                    if not(point[0] in hull):
                        print(point[0])
                    # cv2.line(contour, start, end, [127, 127, 127], 2)
                        cv2.circle(contour_masked, point, 5, [0, 0, 200], -1)
                print("Reached")
            except Exception:
                pass

    cv2.imshow("Live", frame)
    cv2.imshow("Mask", contour_masked)

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