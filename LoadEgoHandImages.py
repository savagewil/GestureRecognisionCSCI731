from glob import glob
import scipy.io
import cv2

path = "D:\\Datasets\\EGOHANDS\\_LABELLED_SAMPLES"

directories = glob("%s/*/" % path)
BROKE = False
for dir in directories:

    mat = scipy.io.loadmat('%s/polygons.mat'%dir)
    images = glob("%s/*.jpg" % dir)
    for image_index in range(len(images)):
        print(images[image_index])
        im = cv2.imread(images[image_index])
        for hand_poly in mat["polygons"][0, image_index]:
            if hand_poly.shape[1] > 0:
                hand_poly = hand_poly.reshape((-1, 1, 2)).astype(int)
                cv2.drawContours(im, [hand_poly], 0, (0, 200, 0), 3)
        print(im)
        cv2.imshow("image", im)
        k = cv2.waitKey(100)
        if k % 256 == 27:
            # ESC pressed
            BROKE = True
            print("Escape hit, closing...")
            break
    if BROKE:
        break
    print(dir, images)
    print(mat["polygons"][0, 0][0].shape)