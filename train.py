from image_gen import ImageSequence
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Reshape, LeakyReLU
from keras import backend as K
from keras import optimizers, metrics
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from models.simple_tunnel import DeepModel2


config = tf.ConfigProto(intra_op_parallelism_threads=7,\
                inter_op_parallelism_threads=7, allow_soft_placement=True,\
                        device_count = {'CPU' : 7, 'GPU' : 1})
session = tf.Session(config=config)
K.set_session(session)


deepModel = DeepModel2(64, 8).model

sgd = optimizers.SGD(lr=0.007, decay=0.0, momentum=0.05, nesterov=True, clipnorm=1.0)
deepModel.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['mean_squared_error'])


MODEL_WEIGHTS_FILE = 'flow_simple_tunnel.h5'



def schedule(epoch, lr=0.01):
    lr_max = 0.01
    lr_min = 0.00001
    return lr_min + (0.6 ** (epoch % 10)) * lr_max

callbacks = [EarlyStopping('val_loss', patience=5),
             LearningRateScheduler(schedule, verbose=1),
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]

gen = ImageSequence(8, (20, 20), (5, 5), maxlen=64)
test_gen = ImageSequence(8, (20, 20), (5, 5), maxlen=16)
with tf.device('/gpu:0'):
   history = deepModel.fit_generator(generator=gen,
                                     validation_data=test_gen,
                                     callbacks=callbacks,
                                     use_multiprocessing=True,
                                     workers=4,
                                     epochs=35)

deepModel.save('simple_tunnel_scr')
