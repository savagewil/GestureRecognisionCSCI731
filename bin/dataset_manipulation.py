import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2


class KaggleHandDetectionDataset(Dataset):
    """Custom loader for the Kaggle Hand Detection Dataset"""

    def __init__(self, csv_file, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with image filepaths and gt classes
            transforms (callable, optional): Optional transforms to be applied on a sample
        """
        self.images_frame = pd.read_csv(csv_file)
        # print(self.images_frame)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_frame.iloc[idx, 0]
        img = cv2.imread(img_name)
        y = int(self.images_frame.iloc[idx, 1]) - 1
        sample = {'image': img, 'y': y}

        if len(self.transforms) > 0:
            for _, transform in enumerate(self.transforms):
                sample = transform(sample)
        return sample


class Rescale(object):
    """Used to rescale an image to a given size. Useful for the CNN

    Args:
        output_size (tuple or int): Desired output size after rescaling. If tuple, output is matched to output_size.
        If int smaller of width/height is matched to output_size, keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        y = sample['y']
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h // w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w // h
        else:
            new_h, new_w = self.output_size

        img = cv2.resize(image, (new_h, new_w))
        # print(f"after resize: {img.shape}")
        return {'image': img, 'y': y}


class Recolor(object):
    """Used to recolor an image using cv2

    Args:
        flag (cv2.COLOR_): color to swap to
    """

    def __init__(self, color):
        self.color = color

    def __call__(self, sample):
        y = sample['y']
        image = sample['image']
        # print(f"before recolor: {image.shape}")

        # cvtColor to gray drops the damned channel dimension but we need it
        img_cvt = cv2.cvtColor(image, self.color)
        # fucking hack an extra dim to appease pytorch's bitchass
        img_cvt = np.expand_dims(img_cvt, axis=-1)
        # print(f"after exansion: {img_cvt.shape}")
        return {'image': img_cvt, 'y': y}


class ToTensor(object):
    """Convert ndarrays to pytorch Tensors"""

    def __call__(self, sample):
        y = sample['y']
        image = sample['image']

        # swap color axis b/c
        # numpy img: H x W x C
        # torch img: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'y': y}


def reshape_img(image, output_size):
    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h // w, output_size
        else:
            new_h, new_w = output_size, output_size * w // h
    else:
        new_h, new_w = output_size

    img = cv2.resize(image, (new_h, new_w))
    return img