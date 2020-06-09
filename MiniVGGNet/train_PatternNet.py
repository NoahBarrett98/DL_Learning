"""
Training for minivgg using PatternNet dataset

"""

from tensorflow.keras.optimizers import SGD
from MiniVGGNet.MiniVGG import MiniVGGNetModel
import util
from load.NPYDataGenerator import NPYDataGenerator

lr = 1e-2
BATCH_SIZE = 128
NUM_EPOCHS = 60

# instantiate model
model = MiniVGGNetModel(len(util.PNET_LABELS))

# initialize the optimizer and compile the model
opt = SGD(lr=lr, momentum=0.9, decay=lr / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# load data
training_generator = NPYDataGenerator(file_dir=r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TRAIN\NPY",
                                      labels=util.PNET_LABELS)
validation_generator = NPYDataGenerator(file_dir=r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TEST\NPY",
                                        labels=util.PNET_LABELS)

history = model.fit(training_generator,
                    epochs=NUM_EPOCHS )
