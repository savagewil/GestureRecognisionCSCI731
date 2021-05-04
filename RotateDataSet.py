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

    for angle in range(0, 360, 20):
        file_parts = os.path.split(file)
        newpath = out_path  + file_parts[1][:3] + f'{angle:02d}' + file_parts[1][3:]
        print(newpath)
        rotated = rotate_image(image, angle)
        # cv2.imshow("rotated", rotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # print(rotated.shape)
        # break
        cv2.imwrite(newpath, rotated)
