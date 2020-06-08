"""

Implementation of ResNet
Dataset: CIFAR 10

resources:  K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.
            arXiv preprint arXiv:1512.03385,2015.
            https://keras.io/
            https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_06_3_resnet.ipynb
            https://en.wikipedia.org/wiki/Residual_neural_network

resnet is a ANN that builds on constructs known from pyramidal cells in the cerebral cortex
"""

import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import matplotlib.pyplot as plt
from six.moves import cPickle
import time

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def show():
    x = x_train.astype("uint8")

    fig, axes1 = plt.subplots(10,10,figsize=(10,10))
    for j in range(10):
        for k in range(10):
            i = np.random.choice(range(len(x)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(x[i:i+1][0])
    plt.show()

# show()

###################
# Training params #
# based on paper  #
###################
batch_size = 32
epochs = 200
num_classes = np.unique(y_train).shape[0]
colours = x_train.shape[3]
initial_lr = 1e-3

"""
There are two versions of resnet:
ResNet v2 is an improved version of ResNet v1
"""
version = 2

# depth is computed from supplied model param
if version == 1:
    depth = colours * 6 + 2
elif version == 2:
    depth = colours * 9 + 2

################################
# learning rate decay schedule #
################################
def lr_schedule(epoch, num_epochs=epochs, lr=initial_lr):
    """
    Scheduler based on paper,
    for the setting used in paper: scheduler will differ based on num epochs
    matching original changes occurring at 180, 160, 120, 80
    :param epoch: current epoch
    :param num_epochs:number of epochs
    :param lr: initial learning rate
    :return: calculated learning rate
    """
    # based on paper: 200 epochs
    # alternative amount schedules on a
    # percentage based schedule rather than
    # the hard coded differences
    if epoch > 0.90 * num_epochs:
        lr *= 0.5e-3
    elif epoch > 0.80 * num_epochs:
        lr *= 1e-3
    elif epoch > 0.60 * num_epochs:
        lr *= 1e-2
    elif epoch > 0.40 * num_epochs:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr
############
# ResBlock #
############
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """
    resnet layer as implemented in paper
    :param inputs: tensor activation from previous layer
    :param num_filters: int number of filters for this layer
    :param kernel_size: int dimensions for kernel for conv2d
    :param strides: int dimension of stride for kernel
    :param activation: name of activation function (typically ReLu)
    :param batch_normalization: bool to toggle batch activation
    :param conv_first: bool to toggle where conv lays in architecture
    :return: tensor output for next layer
    """
    # intialize conv layer #
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    # forward pass #
    x = inputs
    if conv_first:
        # conv is called first #
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x




def resnet_v1(input_shape, depth, num_classes=10):
    """
    ResNet V1 Implementation:
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature
    map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of
    filters is
    doubled. Within each stage, the layers have the same number
    filters and the same number of filters.

    Source:
    K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition.
    arXiv preprint arXiv:1512.03385,2015.

    :param input_shape: tensor shape of input
    :param depth: int number of convolutional layers
    :param num_classes: number of classes ( for output layer of net )
    :return: returns keras model
    """
    num_filters = 16
    num_res_blocks = int((depth-2)/6)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units #
    # 3 layers
    # stage 0: 32x32, 16 filters
    # stage 1: 16x16, 32 filters
    # stage 2:  8x8,  64 filters
    for stack in range(3):
        # iterate through each resblock in 64, 128, 256, 512
        for res_block in range(num_res_blocks):
            # no intial downsampling
            strides = 1

            # first layer but not first stack
            if stack > 0 and res_block == 0:
                strides = 2  # downsample by factor of 2

            # first feed, uses activation
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            # no activation
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)

            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            # incorporating residual by including y and x
            x = tensorflow.keras.layers.add([x, y])
            # output with activation
            x = Activation('relu')(x)

        # increment num_filters by 2* i.e. 16->32
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU

     # avg pool after last conv layer
    x = AveragePooling2D(pool_size=8)(x)

    # fully connected layer
    y = Flatten()(x)

    # classification output softmax
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v2(input_shape, depth, num_classes=10):
    """
    second version of resnet: ResNet v2

    Source:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings in deep residual networks.
    In European conference on computer vision (pp. 630-645). Springer, Cham
    :param input_shape: tensor shape of input
    :param depth: int number of convolutional layers
    :param num_classes:int number of classes in dataset
    :return: keras model
    """
    # intial features
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)
    # iter through stages
    for stage in range(3):

        for res_block in range(num_res_blocks):
            activation = 'relu'
            # v2 uses batch normalization
            batch_normalization = True
            # intialize with no downsampling
            strides = 1
            if stage == 0:
                # next layer has x4 filters than first
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                # after first layer, filters growing by factor of two
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            # skip connection addition
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # avg pool same as v1
    x = AveragePooling2D(pool_size=8)(x)
    # fully connected layer
    y = Flatten()(x)

    # output as distribution (softmax)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

################
# create model #
################
SUBTRACT_PIXEL_MEAN = True
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# subtract pixel mean is enabled
if SUBTRACT_PIXEL_MEAN:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# Create the neural network
if version == 1:
    model = resnet_v1(input_shape=input_shape, depth=depth)

else:
    model = resnet_v2(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

model.summary()

#####################
# training function #
#####################
USE_AUGMENTATION=True

start_time = time.time()

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not USE_AUGMENTATION:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
                                # set input mean to 0 over the dataset
                                featurewise_center=False,
                                # set each sample mean to 0
                                samplewise_center=False,
                                # divide inputs by std of dataset
                                featurewise_std_normalization=False,
                                # divide each input by its std
                                samplewise_std_normalization=False,
                                # apply ZCA whitening
                                zca_whitening=False,
                                # epsilon for ZCA whitening
                                zca_epsilon=1e-06,
                                # randomly rotate images in the range (deg 0 to 180)
                                rotation_range=0,
                                # randomly shift images horizontally
                                width_shift_range=0.1,
                                # randomly shift images vertically
                                height_shift_range=0.1,
                                # set range for random shear
                                shear_range=0.,
                                # set range for random zoom
                                zoom_range=0.,
                                # set range for random channel shifts
                                channel_shift_range=0.,
                                # set mode for filling points outside the input boundaries
                                fill_mode='nearest',
                                # value used for fill_mode = "constant"
                                cval=0.,
                                # randomly flip images
                                horizontal_flip=True,
                                # randomly flip images
                                vertical_flip=False,
                                # set rescaling factor (applied before any other transformation)
                                rescale=None,
                                # set function that will be applied on each input
                                preprocessing_function=None,
                                # image data format, either "channels_first" or "channels_last"
                                data_format=None,
                                # fraction of images reserved for validation
                                # (strictly between 0 and 1)
                                validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=0, workers=1,
                        callbacks=callbacks,
                        use_multiprocessing=False)

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(time.hms_string(elapsed_time)))

