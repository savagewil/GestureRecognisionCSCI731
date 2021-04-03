from glob import glob
import random


SAVE_FILE = "D:\Datasets\SKIN\skin_notskin_samples.txt"
PROGRESS_FILE = "D:\Datasets\SKIN\location.txt"
import scipy.io
import cv2
import numpy as np

path = "D:\\Datasets\\EGOHANDS\\_LABELLED_SAMPLES"

BUFFER = 200

directories = glob("%s/*/" % path)
BROKE = False
for dir in directories:
    
    mat = scipy.io.loadmat('%s/polygons.mat'%dir)
    images = glob("%s/*.jpg" % dir)
    for image_index in range(len(images)):

        im = cv2.imread(images[image_index])
        mask_me = np.zeros_like(im)
        mask_you = np.zeros_like(im)
        for hand_poly_index in range(len(mat["polygons"][0, image_index])):
            hand_poly = mat["polygons"][0, image_index][hand_poly_index]
            if hand_poly_index < 2:
                if hand_poly.shape[1] > 0:
                    hand_poly = hand_poly.reshape((-1, 1, 2)).astype(int)
                    cv2.drawContours(mask_me, [hand_poly], 0, (255, 0, 0), thickness=-1)
            else:
                if hand_poly.shape[1] > 0:
                    hand_poly = hand_poly.reshape((-1, 1, 2)).astype(int)
                    cv2.drawContours(mask_you, [hand_poly], 0, (255, 0, 0), thickness=-1)

        mask_me = mask_me[..., 0] > 127
        mask_you = mask_you[..., 0] > 127


        Xs = np.arange(0, mask_you.shape[1])
        Ys = np.arange(0, mask_you.shape[0])
        Xmesh, Ymesh = np.meshgrid(Xs, Ys)
        #
        # Xmin = max(0, np.min(Xs[np.nonzero(np.any(mask_you, axis=0))]) - BUFFER)
        # Xmax = min(mask_you.shape[1], np.max(Xs[np.nonzero(np.any(mask_you, axis=0))]) + BUFFER)
        # print(np.min(Ys[np.nonzero(np.any(mask, axis=1))]))
        # print(np.max(Ys[np.nonzero(np.any(mask, axis=1))]))
        # print(np.min(np.any(mask, axis=1) * Ys))

        mask = mask_you | mask_me
        mask_draw = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        count = np.sum(mask)

        mask_not = (~mask).flatten()
        Xmesh = Xmesh.flatten()
        Ymesh = Ymesh.flatten()
        y_points = Ymesh[mask_not]
        x_points = Xmesh[mask_not]
        index = np.random.choice(y_points.shape[0], count, replace=False)
        mask_other = np.zeros_like(mask)

        mask_other[(y_points[index]), (x_points[index])] = True
        mask_other_draw = np.reshape(mask_other, (mask_other.shape[0], mask_other.shape[1], 1))

        # print(count)

        cv2.imshow("image", im)
        cv2.imshow("mask", mask_draw * im)
        cv2.imshow("mask other", mask_other_draw * im)

        k = cv2.waitKey(100)
        if k % 256 == 27:
            # ESC pressed
            BROKE = True
            print("Escape hit, closing...")
            break
    if BROKE:
        break