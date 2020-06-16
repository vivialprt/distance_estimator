"""This file contain all necessary tools
for loading dataset from corresponding files.
"""
import json
import os
from typing import List, Union, Tuple, Any

from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from albumentations.augmentations.transforms import Blur, RandomCrop, \
    HueSaturationValue
import numpy as np
import cv2


def get_train_val(
    labels_path: str,
    test_size: float
) -> Tuple[List[Union[np.ndarray, int]], ...]:
    """
    Loads labels and image names
    and splits them in train and val samples.
    """
    assert os.path.exists(labels_path), 'No such file: {}'.format(labels_path)
    with open(labels_path) as labels_file:
        raw_data = json.load(labels_file)['data']
    filenames = []
    labels = []
    for item in raw_data:
        if item['mult'] == 1:  # only working with unscaled images for now
            filenames.append(item['name'])
            labels.append(item['dist'])
    return train_test_split(filenames, labels, test_size=test_size)


class BatchGenerator(Sequence):
    """Used in model fitting with generator."""

    def __init__(
        self,
        im_dir: str,
        image_filenames: List[Any],
        labels: List[Any],
        grayscale: bool,
        im_shape: Tuple[int, int],
        batch_size: int
    ):
        """
        :param im_dir: directory with images
        :param image_filenames: filenames of target images
        :param labels: labels for images
        :param grayscale: whether to grayscale images or not
        :param im_shape: shape to resize output images to
        :param batch_size: batch size
        """
        assert os.path.exists(im_dir), 'Specified directory does not exist: {}'.format(im_dir)
        self._image_filenames: List[str] = [
            os.path.join(im_dir, im_name) for im_name in image_filenames
        ]
        self._labels: List[float] = labels
        self._batch_size: int = batch_size
        self._grayscale: bool = grayscale
        self._im_shape: Tuple[int, int] = im_shape

    def _im_preproc(self, image: np.ndarray) -> np.ndarray:
        """
        Perform image preprocessing and augmenting.
        :param image: raw image
        :return: processed image
        """
        image = cv2.resize(image, tuple(self._im_shape))
        blur = Blur(p=0.9, blur_limit=7)
        hsv = HueSaturationValue(p=0.9)
        crop = RandomCrop(p=1, height=self._im_shape[1], width=self._im_shape[0])

        image = blur.apply(image, **blur.get_params())
        image = hsv.apply(image, **hsv.get_params())
        image = crop.apply(image, **crop.get_params())
        if self._grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image[:, :, np.newaxis]
        return image

    def __len__(self):
        """
        Returns total num of batches.
        """
        return int(np.ceil(len(self._image_filenames) / float(self._batch_size)))

    def __iter__(self):
        return self

    def __getitem__(self, idx: int) -> np.array:
        batch_pos = idx * self._batch_size
        batch_image_names = self._image_filenames[batch_pos:batch_pos + self._batch_size]
        batch_x = [self._im_preproc(cv2.imread(file_name)) for file_name in batch_image_names]
        batch_y = self._labels[batch_pos:batch_pos + self._batch_size]

        return np.array(batch_x), np.array(batch_y)

    def __next__(self):
        for idx in range(len(self)):
            yield self[idx]
