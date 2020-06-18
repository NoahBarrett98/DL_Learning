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
data, info = tfds.load('eurosat', split="train", with_info=True, batch_size=-1)
ds_size = info.splits["train"].num_examples
num_features = info.features["label"].num_classes
data = tfds.as_numpy(data)
train_data, train_label = data["image"], data["label"]

logdir = "Tensorboard/logs/simpleCV/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

### build conv model ###
conv = tf.keras.models.Sequential()
conv.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
conv.add(tf.keras.layers.MaxPooling2D((2, 2)))
conv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
conv.add(tf.keras.layers.MaxPooling2D((2, 2)))
conv.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

### flatten out for classification output
conv.add(tf.keras.layers.Flatten())
conv.add(tf.keras.layers.Dense(64, activation='relu'))
conv.add(tf.keras.layers.Dense(10))

### view architecture ###
conv.summary()


### train and compile ###
conv.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

training_history = conv.fit(
                            train_data,
                            train_label,
                            batch_size=batch_size,
                            verbose=0, # Suppress chatty output; use Tensorboard instead
                            epochs=EPOCHS,
                            callbacks=[tensorboard_callback],
                        )