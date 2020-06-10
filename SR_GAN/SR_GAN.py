"""
Super-resolution generative adversarial network
applies a deep network in combination with an adversarial network
GAN upsamples a low res image to super resolution images (LR->SR)

following design from: Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
                        Network  https://arxiv.org/pdf/1609.04802.pdf

composes of convolution layers, batch normalization and parameterized ReLU (PRelU)

loss functions, comprises of reconstruction loss and adversarial loss:
    -uses perceptual loss, measuring MSE of features extracted by a VGG-19 network
        ->for a specific layer, we want their features to be matched st MSE is minimized
    -discriminator is trained using the typical GAN discriminator loss

"Goal is to train a generating function G that estimates for a given LR input image,
its corresponding HR counterpart."

general idea: train a generative model G with the goal of fooling a differentiable
discriminator D that is tained to distinguish super-resolved images from real images
"""

from tensorflow.keras.models import Model
from Generator import Generator
from Discriminator import Discriminator

class SR_GAN(Model):
    def __init__(self,
                 residual_blocks=16,
                 momentum=0.8,
                 leakyrelu_alpha=0.2):
        # call the parent constructor
        super(SR_GAN, self).__init__()

        #############
        # Generator #
        #############
        self.generator = Generator()

        #################
        # Discriminator #
        #################
        self.discriminator = Discriminator()
        self.discriminator.trainable = False

    def call(self, inputs):

        #############
        # Generator #
        #############
        x = Generator.call(inputs)
        #################
        # Discriminator #
        #################
        return Discriminator.call(x)




