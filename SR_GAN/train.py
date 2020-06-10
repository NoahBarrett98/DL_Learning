from SR_GAN import SR_GAN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import time
from MiniVGGNet.MiniVGG import MiniVGGNetModel
from Generator import Generator
from Discriminator import Discriminator
from tensorflow.keras import Input
from load.NPYDataGenerator import NPYDataGeneratorSR
####################
# Hyper-Parameters #
####################
epochs = 20000
batch_size = 1
num_outputs = 1

# build VGG net #
# instantiate model
vgg = MiniVGGNetModel(num_outputs)

# initialize the optimizer and compile the models
opt = Adam(0.0002, 0.5)

vgg.compile(loss="mse", optimizer=opt,
	metrics=["accuracy"])

kwargs = {"name":"discriminator"}
# instantiate discriminator
discriminator = Discriminator(**kwargs)
discriminator.compile(loss='mse', optimizer=opt,
	metrics=["accuracy"])

kwargs = {"name":"generator"}
# instantiate generator
generator = Generator(**kwargs)
generator.compile(loss='mse', optimizer=opt,
	metrics=["accuracy"])

#######################
# Adversarial Network #
#######################
# Shape of low-resolution and high-resolution images #
low_resolution_shape = (64, 64, 3)
high_resolution_shape = (256, 256, 3)

input_high_resolution = Input(shape=high_resolution_shape)
input_low_resolution = Input(shape=low_resolution_shape)



generated_high_resolution_images = generator(input_low_resolution)
features = vgg(generated_high_resolution_images)
discriminator.trainable = False
probs = discriminator(generated_high_resolution_images)

adversarial_model = Model([input_low_resolution, input_high_resolution], [probs, features])
adversarial_model.compile(loss=['binary_crossentropy', 'mse'],
            loss_weights=[1e-3, 1], optimizer=opt)

### KEEPING TRACK ###
tensorboard = TensorBoard(log_dir="logs/".format(time.time()))
tensorboard.set_model(generator)
tensorboard.set_model(discriminator)


### data ###
training_generator = NPYDataGeneratorSR(file_dir=r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TRAIN\super_res_NPY"
                                      )
validation_generator = NPYDataGeneratorSR(file_dir=r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TEST\super_res_NPY"
                                        )

# train the network
History = adversarial_model.fit_generator(training_generator,
								epochs=20000,
								validation_data=validation_generator,
								verbose=1
								)

"""History = adversarial_model.fit_generator(training_generator,
								epochs=20000,
								batch_size=batch_size,
								validation_data=validation_generator,
								verbose=1
								)"""
