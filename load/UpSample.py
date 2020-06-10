import numpy as np
import scipy.ndimage
import util
from load.NPYDataGenerator import NPYDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import random
import glob
import os
from tqdm import tqdm


############################
# build patternNet dataset #
############################


def make_train_data(train_data, train_dir):
    print("making train sets...")
    data = iter(train_data)
    for i, batch in tqdm(enumerate(data)):

        print(batch[0].shape)
        for j, val in enumerate(batch[0]):
            img = Image.fromarray(np.uint8(val))
            # pair[1] = img.resize([64,64])
            fname = train_dir + "\pair_{}.npy".format(i+j)

            with open(fname, "wb") as f:
                np.save(f, img)
                np.save(f, img.resize([64,64]))

def make_train_data(test_data, test_dir):
    print("making test sets...")
    data = iter(test_data)
    for i, batch in tqdm(enumerate(data)):
        for j, val in enumerate(batch[0]):
            img = Image.fromarray(np.uint8(val))
            # pair[1] = img.resize([64,64])
            fname = test_dir + "\pair_{}.npy".format(i+j)

            with open(fname, "wb") as f:
                np.save(f, img)
                np.save(f, img.resize([64,64]))

def show_sample(train_dir, test_dir):
    train_sample = glob.glob(train_dir+"\*")
    train_sample = train_sample[random.randint(0, len(train_sample))]
    test_sample = glob.glob(test_dir + "\*")
    test_sample = test_sample[random.randint(0, len(test_sample))]
    fig, ax = plt.subplots(2,2)
    with open(train_sample, "rb") as f:
        hr = np.load(f)
        lr = np.load(f)
        ax[0,0].imshow(hr)
        ax[0,1].imshow(lr)
        ax[0,0].set_title("hr, train")
        ax[0,1].set_title("lr, train")

    with open(test_sample, "rb") as f:
        hr = np.load(f)
        lr = np.load(f)
        ax[1,0].imshow(hr)
        ax[1,1].imshow(lr)
        ax[1,0].set_title("hr, test")
        ax[1,1].set_title("lr, test")

    plt.tight_layout()
    plt.show()

"""usage example"""
train_data = NPYDataGenerator(r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TRAIN\NPY",
                        labels=util.PNET_LABELS)
test_data = NPYDataGenerator(r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TEST\NPY",
                        labels=util.PNET_LABELS)

train_dir = r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TRAIN\super_res_NPY"
test_dir = r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TEST\super_res_NPY"
#make_train_data(test_data, test_dir)

show_sample(train_dir, test_dir)
