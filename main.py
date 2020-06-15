"""Perform model training and evaluating"""

import sys
import os
import argparse
from model import create_model
from datetime import datetime
from dataset import get_train_val, BatchGenerator
from yaml import load
from easydict import EasyDict
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks.tensorboard_v1 import TensorBoard
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.models import load_model


if __name__ == '__main__':
    with open('config.yml') as config:
        CFG = EasyDict(load(config))
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    TRAIN_LOG_DIR = os.path.join(CFG.TRAIN.LOGS_PATH, TIMESTAMP)
    TEST_LOG_DIR = os.path.join(CFG.TEST.LOGS_PATH, TIMESTAMP)
    WEIGHTS_DIR = os.path.join(CFG.TRAIN.SAVE_PATH, TIMESTAMP)

    # print('[INFO] Creating model...')
    # model = create_model(
    #     width=CFG.ARCH.INPUT_SIZE[0],
    #     height=CFG.ARCH.INPUT_SIZE[1],
    #     filters=CFG.ARCH.FILTERS)
    # opt = Adam(
    #     lr=CFG.TRAIN.LR,
    #     decay=CFG.TRAIN.DECAY)
    # model.compile(
    #     loss=CFG.TRAIN.LOSS,
    #     optimizer=opt)
    # plot_model(
    #     model,
    #     to_file='model.png',
    #     show_shapes=True,
    #     expand_nested=True)

    print('[INFO] Loading model...')
    model = load_model(CFG.TEST.WEIGHTS)

    print('[INFO] Preparing data generators...')
    train_x, test_x, train_y, test_y = get_train_val(
        labels_path=CFG.DATA.DIST_PATH,
        test_size=CFG.DATA.TEST_SIZE
    )
    train_generator = BatchGenerator(
        im_dir=CFG.DATA.IM_DIR,
        image_filenames=train_x,
        labels=train_y,
        grayscale=CFG.DATA.GRAYSCALE,
        im_shape=CFG.ARCH.INPUT_SIZE,
        batch_size=CFG.TRAIN.BATCH_SIZE
    )
    test_generator = BatchGenerator(
        im_dir=CFG.DATA.IM_DIR,
        image_filenames=test_x,
        labels=test_y,
        grayscale=CFG.DATA.GRAYSCALE,
        im_shape=CFG.ARCH.INPUT_SIZE,
        batch_size=CFG.TRAIN.BATCH_SIZE
    )

    # print('[INFO] Training model...')
    # model.fit_generator(
    #     generator=train_generator,
    #     validation_data=test_generator,
    #     epochs=CFG.TRAIN.EPOCHS,
    #     callbacks=[TensorBoard(log_dir=TRAIN_LOG_DIR),
    #                ModelCheckpoint(WEIGHTS_DIR + '{epoch}.h5', period=50)
    #                ])

    print('[INFO] Evaluating model...')
    score = model.evaluate_generator(
        generator=train_generator,
        verbose=1,
        callbacks=[TensorBoard(log_dir=TEST_LOG_DIR)]
    )
    print(score)

    sys.exit(0)
