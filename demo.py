"""Script for demonstrating model performance and accuracy."""
import json
import os
import time

import yaml
import numpy as np
from easydict import EasyDict
import cv2
from keras.models import load_model

if __name__ == '__main__':
    with open('demo_config.yml') as config:
        CFG = EasyDict(yaml.load(config))

    mode = CFG.MODE
    if mode not in ['image']:
        raise NotImplementedError(f'Mode {mode} not supported.')
    print('[INFO] Loading model...')
    model = load_model(CFG.WEIGHTS)
    if mode == 'image':
        with open(CFG.LABELS) as labels:
            data = json.load(labels)['data']
        data = [item for item in data if item['mult'] == 1]
        samples = np.random.choice(data, CFG.TEST_SIZE)
        for sample in samples:
            im_path = os.path.join(CFG.DATA, sample['name'])
            gt_dist = sample['dist']
            image = cv2.imread(im_path)
            background = np.full((300, 300, 3), 255, np.uint8)
            width = int(model.inputs[0].shape[1])
            height = int(model.inputs[0].shape[2])

            image = cv2.resize(image, (width, height))
            if CFG.GRAYSCALE:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image[:, :, np.newaxis]

            start = time.time()
            dist = model.predict(np.array([image]))
            latency = time.time() - start

            cv2.putText(
                img=background,
                text=f'Predicted: {dist[0][0]:.3f}',
                org=(0, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1
            )
            if CFG.SHOW_GT:
                cv2.putText(
                    img=background,
                    text=f'Real: {gt_dist:.3f}',
                    org=(0, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=1
                )
            if CFG.SHOW_LATENCY:
                cv2.putText(
                    img=background,
                    text=f'Latency: {latency:.6f}',
                    org=(0, 45),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=1
                )
            background[100:100 + height, 100:100 + width] = image

            cv2.imshow('result', background)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
