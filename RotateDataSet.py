import cv2
import glob
import os
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

print()
Binary_file_path = "data/asl/asl_data/binary_frames/*.png"
out_path = "data/asl/asl_data/binary_frames_rotated/"

for file in glob.glob(Binary_file_path):
    image = cv2.imread(file)

    for flip in range(2):
        for angle in range(0, 360, 20):
            file_parts = os.path.split(file)
            newpath = out_path + file_parts[1][:3] + f'{angle:03d}{flip}' + file_parts[1][3:]
            print(newpath)
            if flip == 1:
                rotated = rotate_image(cv2.flip(image, 1), angle)
            else:
                rotated = rotate_image(image, angle)

            # cv2.im reak
            # print(rotated.shape)
            # break
            cv2.imwrite(newpath, rotated)
