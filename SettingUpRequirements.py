import numpy
import pandas
import sklearn
import cv2

print(numpy.zeros(10))

print(pandas.DataFrame())

print(sklearn.utils.Path)



cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    frames = []

    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 0])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2])

    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 2])

    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 1])
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 2])

    # frames[1] = numpy.mean(cv2.cvtColor(frames[1], cv2.COLOR_BGR2HSV), axis=1)
    # frames[1] = cv2.cvtColor(frames[1], cv2.COLOR_BGR2HSV)[..., 2]
    # frames[2] = cv2.cvtColor(frames[2], cv2.COLOR_BGR2LAB)[..., 0]
    # frames[3] = cv2.cvtColor(frames[3], cv2.COLOR_BGR2YUV)[..., 0]

    panel = cv2.vconcat([cv2.hconcat([frames[0], frames[1], frames[2]]),
                         cv2.hconcat([frames[3], frames[4], frames[5]]),
                         cv2.hconcat([frames[6], frames[7], frames[8]])])
    cv2.imshow("test", panel)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()