from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Dropout, BatchNormalization

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Reshape, LeakyReLU

class DeepModel2(Sequential):

        def __init__(self, imsize, batch_size):
            inp = Input(shape=(imsize, imsize, 2), dtype='float32')
            conv = Conv2D(64, 7, strides=(1, 1), padding='valid', activation='relu')(BatchNormalization(axis=3)(inp))
            conv = Conv2D(128, 5, strides=(1, 1), padding='valid', activation='relu')(BatchNormalization(axis=3)(conv))
            conv = Conv2D(256, 5, strides=(1, 1), padding='valid', activation='relu')(BatchNormalization(axis=3)(conv))
            conv = Conv2DTranspose(256, 5, strides=(1, 1), padding='valid', activation='relu')(BatchNormalization(axis=3)(conv))
            conv = Conv2DTranspose(128, 5, strides=(1, 1), padding='valid', activation='relu')(BatchNormalization(axis=3)(conv))
            conv = Conv2DTranspose(2, 7, strides=(1, 1), padding='valid')(BatchNormalization(axis=3)(conv))
            final = conv
            self.model = Model(inputs=inp, outputs=final)

        def rate(self, batch):
            return self.model.predict(batch)[0][0]
