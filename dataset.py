"""This file contain all neccessary tools
for loading dataset from corresponding files
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
    """Loads labels and image names
    and splits them in train and val selections.
    """

    if not os.path.exists(labels_path):
        raise RuntimeError('No such file: {}'.format(labels_path))
    with open(labels_path) as labels_file:
        raw_data = json.load(labels_file)['data']
    filenames = []
    labels = []
    for item in raw_data:
        if item['mult'] == 1:
            filenames.append(item['name'])
            labels.append(item['dist'])
    split = train_test_split(filenames, labels, test_size=test_size)
    return split


class BatchGenerator(Sequence):
    """Used in model fitting with generator"""

    def __init__(self,
                 im_dir: str,
                 image_filenames: List[Any],
                 labels: List[Any],
                 grayscale: bool,
                 im_shape: Tuple[int, int],
                 batch_size: int
                 ):
        self.image_filenames: List[str] = [
            os.path.join(im_dir, im_name) for im_name in image_filenames
            ]
        self.labels: List[float] = labels
        self.batch_size: int = batch_size
        self.grayscale: bool = grayscale
        self.im_shape: Tuple[int, int] = im_shape

    def __im_preproc(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, tuple(self.im_shape))
        blur = Blur(p=0.9, blur_limit=7)
        hsv = HueSaturationValue(p=0.9)
        crop = RandomCrop(p=1, height=self.im_shape[1], width=self.im_shape[0])

        image = blur.apply(image, **blur.get_params())
        image = hsv.apply(image, **hsv.get_params())
        image = crop.apply(image, **crop.get_params())
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image[:, :, np.newaxis]

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_pos = idx * self.batch_size
        batch_x = self.image_filenames[batch_pos:batch_pos + self.batch_size]
        batch_y = self.labels[batch_pos:batch_pos + self.batch_size]

        return np.array([
            self.__im_preproc(cv2.imread(file_name))
            for file_name in batch_x]), np.array(batch_y)
