"""This file contains tools for working with network model"""

from typing import List
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Flatten
from keras.layers import Input


def create_model(
        width: int,
        height: int,
        depth: int = 1,
        filters: List[int] = [16, 32, 64]
        ) -> Model:
    """Create cnn model architecture.
    """

    input_shape = (height, width, depth)
    chan_dim = -1
    inputs = Input(shape=input_shape)
    for index, size in enumerate(filters):
        if index == 0:
            x = inputs
        # CONV -> RELU -> BN -> POOL
        x = Conv2D(size, (3, 3),
                   padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # FC -> RELU -> BN -> DROPOUT
    x = Dense(512, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1, kernel_initializer='he_normal')(x)
    x = Activation('linear')(x)

    return Model(inputs, x)
