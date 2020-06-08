import glob
import os
from PIL import Image
import numpy as np
try:
    ROOT_DIR = os.environ["DL_DATA"]
except KeyError:
    os.environ["DL_DATA"] = input("DL data root dir: ")
    ROOT_DIR = os.environ["DL_DATA"]

def save_as_npy(base_dir, fpath):
    new_dir = os.path.join(base_dir, "NPY")
    try:
        os.mkdir(new_dir)
    except:
        pass
    for dir in glob.glob(fpath+"\*"):
        img_class = os.path.basename(dir)
        for i, img in enumerate(glob.glob(dir +"\*.jpg")):
            new_f = os.path.join(new_dir, img_class+"_{}".format(i)+".npy")
            img = Image.open(img)
            with open(new_f, 'wb') as f:
                np.save(f, np.array(img))



def class_to_idx(fname, e_dict):

    return e_dict[fname]

def Load_PatternNet_as_NPY(train_dir, test_dir):
    """
    load patternNet
    """
    dirs = glob.glob(test_dir +"\*")
    img_classes = np.array(set([''.join(os.path.basename(f).split("_")[-2]) for f in dirs]))

    #encoding_dict = {img_class:i for i, img_class in enumerate(img_classes)}

    #################
    # Training data #
    #################
    train_data = [[], []]
    for fname in glob.glob(train_dir+"\*.npy"):
        with open(fname, "rb") as f:

            train_data[0].append(np.load(f))

        # parse fname
        p_fname = os.path.basename(fname).split("_")[-2]
        truth = p_fname == img_classes
        train_data[1].append(truth)

    ################
    # Testing data #
    ################
    test_data = [[], []]
    for fname in glob.glob(test_dir + "\*.npy"):
        with open(fname, "r") as f:
            test_data[0].append(np.load(f))

        # parse fname
        p_fname = os.path.basename(fname).split("_")[-2]
        print(p_fname)
        truth = p_fname == img_classes
        test_data[1].append(truth)

    return train_data, test_data

train, test = Load_PatternNet_as_NPY(r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TRAIN\NPY",
                        r"C:\Users\Noah Barrett\Desktop\School\Research 2020\data\deep_learning\PatternNet\TEST\NPY")