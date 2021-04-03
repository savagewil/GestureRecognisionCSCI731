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
    (thresh, hand_mask) = cv2.threshold(grayImage, 80, 255, cv2.THRESH_BINARY)

    # hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

    contours, _ = cv2.findContours(hand_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_masked = hand_mask.copy()
    # print(contours)
    hands = []
    for contour in contours:
        if 5000 <= cv2.contourArea(contour):
            cv2.drawContours(frame, [contour], 0, (127, 127, 127), 3)
            hull_shape = cv2.convexHull(contour, returnPoints=True)
            # print(hull_shape)
            hull = cv2.convexHull(contour, returnPoints=False)

            cv2.drawContours(frame, [hull_shape], 0, (100, 100, 100), 3)
            try:
                # defects = cv2.convexityDefects(contour, hull)
                for point in contour:
                    if not(point[0] in hull):
                        # print(point[0])
                        # cv2.line(contour, start, end, [127, 127, 127], 2)
                        cv2.circle(frame, point, 5, [0, 0, 200], -1)
                print("Reached")
            except Exception:
                pass



            try:
                defects = cv2.convexityDefects(contour, hull)
                for point in defects:
                    # print(point[0, 3])
                    if point[0, 3] > 500:
                        defect = tuple(contour[point[0, 2]][0])
                        start = tuple(contour[point[0, 0]][0])
                        end = tuple(contour[point[0, 1]][0])
                        dist = np.sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
                        print(f"norm{dist}")
                        if dist > 3:
                            # print(defect)
                            cv2.line(frame, start, end, [255, 0, 0], 5)
                            cv2.circle(frame, defect, 5, [0, 0, 255], -1)
            except Exception as e:
                print("defects failed")
                print(e)
                # print(hull)
            hands.append(cv2.boundingRect(contour))

    cv2.imshow("Live", frame)
    cv2.imshow("Mask", contour_masked)
    for hand_index in range(len(hands)):
        x, y, w, h = hands[hand_index]
        cv2.imshow("Mask%d"%hand_index, frame[y:y+h, x:x+w])

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