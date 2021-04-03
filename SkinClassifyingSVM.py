import numpy as np
import pandas
import sklearn
import cv2

BIAS = [0.1, 0.0, 0.0]
# THRESHOLD = [1.2115692504365638 + BIAS, 2.3111211649131134 + BIAS, 2.350533297981966 + BIAS]
#
# MODEL = [[10.658074283804243, 8.686044065761761],
#          [116.06187695393146, 34.6242892623335],
#          [203.99193849662794, 37.69660241077398]]
THRESHOLD = [2.3410172411581796 + BIAS[0], 3.8690052413348233 + BIAS[1], 2.3914482466047233 + BIAS[2]]

MODEL = [
    [166.90121709038715, 34.588677506339295],
    [146.01930828368626, 3.8062442597457693],
    [154.02795965315875, 9.633518390359571]
]
COLOR_SPACE = cv2.COLOR_BGR2LAB
def get_skin_masks(im, THRESHOLD, MODEL, COLOR, KERNAL_SIZE=11):
    color_frame = cv2.cvtColor(np.uint8(im), COLOR)
    skin_ims = []
    for i in range(3):
        channel_im = color_frame[..., i]
        channel_im_mask = np.abs((channel_im - MODEL[i][0]) / MODEL[i][1]) <= THRESHOLD[i]

        # print(h_frame_mask)
        # channel_im_mask = cv2.morphologyEx(channel_im_mask.astype(np.uint8), cv2.MORPH_OPEN,
        #                                    np.ones((KERNAL_SIZE, KERNAL_SIZE), np.uint8))
        skin_ims.append(np.reshape(channel_im_mask,
                                     (im.shape[0], im.shape[1], 1)))
    return skin_ims


cam = cv2.VideoCapture(0)
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    r_frame = frame.copy()
    r_frame[..., 0] = frame[..., 2]
    r_frame[..., 1] = frame[..., 1]
    r_frame[..., 2] = frame[..., 0]

    c0_frame_mask, c1_frame_mask, c2_frame_mask = get_skin_masks(frame, THRESHOLD, MODEL, COLOR_SPACE)
    # hsv_frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2HSV)
    # h_frame = hsv_frame[..., 0]
    # h_frame_mask = np.reshape(np.abs((h_frame - MODEL[0][0])/MODEL[0][1]) <= THRESHOLD[0],
    #                           (frame.shape[0], frame.shape[1], 1))
    # # print(h_frame_mask)
    # new_frame = frame * h_frame_mask

    # frames[2] = cv2.cvtColor(frames[2], cv2.COLOR_BGR2LAB)[..., 0]
    # frames[3] = cv2.cvtColor(frames[3], cv2.COLOR_BGR2YUV)[..., 0]

    skin_panel = cv2.vconcat([
                    # cv2.hconcat([c0_frame_mask * frame, np.uint8(np.repeat(c0_frame_mask, 3, 2)) * 128]),
                    cv2.hconcat([c1_frame_mask * frame, np.uint8(np.repeat(c1_frame_mask, 3, 2)) * 128]),
                    cv2.hconcat([c2_frame_mask * frame, np.uint8(np.repeat(c2_frame_mask, 3, 2)) * 128]),
                    cv2.hconcat([c1_frame_mask * c2_frame_mask * frame, np.uint8(np.repeat(c1_frame_mask * c2_frame_mask, 3, 2)) * 128])
    ])
    # cv2.imshow("Skin C0?", c0_frame_mask * frame)
    # cv2.imshow("Skin C1?", c1_frame_mask * frame)
    # cv2.imshow("Skin C2?", c2_frame_mask * frame)
    # cv2.imshow("Skin C2&C0?", c0_frame_mask * c2_frame_mask * frame)
    cv2.imshow("Live", frame)
    cv2.imshow("Skin?", skin_panel)
    # cv2.imshow("skin?", new_frame)

    # cv2.imshow("Threshold C0", np.uint8(c0_frame_mask[..., 0]) * 128)
    # cv2.imshow("Threshold C1", np.uint8(c1_frame_mask[..., 0]) * 128)
    # cv2.imshow("Threshold C2", np.uint8(c2_frame_mask[..., 0]) * 128)
    # cv2.imshow("Threshold C2&C0", np.uint8(c2_frame_mask[..., 0] * c0_frame_mask[..., 0]) * 128)


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