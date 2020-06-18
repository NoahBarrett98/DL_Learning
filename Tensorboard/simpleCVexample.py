"""
Simple computer vision application to show the use
of tensorboard using model.fit method

in iPython
"""
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

batch_size = 10
EPOCHS = 100
# load train
data, info = tfds.load('eurosat', split="train", with_info=True)
ds_size = info.splits["train"].num_examples
num_features = info.features["label"].num_classes
train_data = data.batch(batch_size).repeat(EPOCHS)
train_data = tfds.as_numpy(train_data)

logdir = "Tensorboard/logs/simpleCV/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

### build conv model ###
conv = tf.keras.models.Sequential()
conv.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
conv.add(tf.keras.layers.MaxPooling2D((2, 2)))
conv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
conv.add(tf.keras.layers.MaxPooling2D((2, 2)))
conv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

### flatten out for classification output
conv.add(tf.keras.layers.Flatten())
conv.add(tf.keras.layers.Dense(64, activation='relu'))
conv.add(tf.keras.layers.Dense(num_features))

### view architecture ###
conv.summary()


### train and compile ###
conv.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              verbose=0,
              epochs=EPOCHS,
              metrics=['accuracy'],
              callbacks=[tensorboard_callback])