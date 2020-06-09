from SR_GAN.SR_GAN import SR_GAN
from tensorflow.keras.optimizers import Adam

# instantiate model
model = SR_GAN()

# initialize the optimizer and compile the model
opt = Adam(0.0002, 0.5)
print("[INFO] training network...")
model.compile(loss='mse', optimizer=opt,
	metrics=["accuracy"])