"""This file contain all neccessary tools
for loading dataset from corresponding files
"""

from sklearn.model_selection import train_test_split
from typing import List, Dict, Union, Tuple
import numpy as np
import json
import os
import cv2


IMAGE_FORMAT: str = '.png'


def load_images(im_dir: str) -> Dict[str, np.ndarray]:
    """Function that simply loads images from a specified directory
    and returns them as a list
    """

    if not os.path.exists(im_dir):
        raise RuntimeError('No such directory: {}'.format(im_dir))
    filenames = [name for name in os.listdir(im_dir)
                 if name.endswith(IMAGE_FORMAT)]
    if len(filenames) < 1:
        raise RuntimeError('No images in specified directory.')
    im_dict = {}
    for filename in filenames:
        full_name = os.path.join(im_dir, filename)
        im_dict[filename] = cv2.imread(full_name)
    return im_dict


def load_distance(path: str) -> Dict[str, float]:
    """Function that loads distance from a json file"""

    if not os.path.exists(path):
        raise RuntimeError('No such file: {}'.format(path))
    with open(path) as labels_file:
        raw_data = json.load(labels_file)['data']
    data = {}
    for item in raw_data:
        data[item['name']] = item['dist'] * item['mult']
    return data


def prepare_data(
    images: Dict[str, np.ndarray],
    dists: Dict[str, float]
        ) -> Tuple[List[Union[np.ndarray, int]], ...]:
    """Prepairs all neccessary data and returns in format
    (train_im, train_dist, test_im, test_dist)
    """

    all_images = []
    all_dists = []
    for name in images.keys():
        if name in dists.keys():
            all_images.append(images[name])
            all_dists.append(dists[name])
    return train_test_split(all_images, all_dists, test_size=0.25)
