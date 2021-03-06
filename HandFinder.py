import cv2
import numpy as np
class HandFinder:
    def __init__(self):
        pass

    def extractHands(self, frame):
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, hand_mask) = cv2.threshold(grayImage, 80, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(hand_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()

        hands = []
        hand_rect = []
        for contour in contours:
            if 2000 <= cv2.contourArea(contour):
                cv2.drawContours(frame_copy, [contour], 0, (127, 127, 127), 3)
                hull_shape = cv2.convexHull(contour, returnPoints=True)

                hull = cv2.convexHull(contour, returnPoints=False)

                cv2.drawContours(frame_copy, [hull_shape], 0, (100, 100, 100), 3)
                try:

                    for point in contour:
                        if not (point[0] in hull):

                            cv2.circle(frame_copy, point, 5, [0, 0, 200], -1)
                    print("Reached")
                except Exception:
                    pass

                try:
                    defects = cv2.convexityDefects(contour, hull)
                    for point in defects:
                        if point[0, 3] > 500:
                            defect = tuple(contour[point[0, 2]][0])
                            start = tuple(contour[point[0, 0]][0])
                            end = tuple(contour[point[0, 1]][0])
                            dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
                            #print(f"norm{dist}")
                            if dist > 3:
                                cv2.line(frame_copy, start, end, [255, 0, 0], 5)
                                cv2.circle(frame_copy, defect, 5, [0, 0, 255], -1)
                except Exception as e:
                    print("defects failed")
                    print(e)
                x,y,w,h = cv2.boundingRect(contour)
                #print("test")
                hand = hand_mask[y:y+h, x:x+w]
                h_margin = int(h * 0.4)
                w_margin = int(w * 0.4)
                hand = cv2.copyMakeBorder(hand, h_margin, h_margin, w_margin, w_margin, cv2.BORDER_CONSTANT, None, [0,0,0])
                hand = cv2.morphologyEx(hand, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (20,20)))
                # hand_back = np.zeros(tuple(np.int(np.array(hand.shape) * np.array([1.2,1.2,1.0]))))
                # hand_back[int(h)]
                hands.append(hand)
                hand_rect.append((x,y,w,h))
        return frame_copy, hands, hand_rect