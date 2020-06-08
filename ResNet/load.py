import util
import glob
import numpy as np
import os
class np_iterable_from_dir():
    def __init__(self, train_dir, test_dir):
        self.train_files = iter(glob.glob(train_dir+"\*"))
        self.test_files = iter(glob.glob(test_dir + "\*"))
        self._img_classes = np.array(set([''.join(os.path.basename(f).split("_")[-2]) for f in glob.glob(test_dir + "\*")]))

    def get_next_test_pair(self):
        fname = next(self.train_files)
        with open(fname, "r") as f:
            x = np.load(f)

        # parse fname
        p_fname = os.path.basename(fname).split("_")[-2]
        y = p_fname == self._img_classes

        return x, y

    def get_next_test_pair(self):
        fname = next(self.test_files)
        with open(fname, "r") as f:
            x = np.load(f)

        # parse fname
        p_fname = os.path.basename(fname).split("_")[-2]
        y = p_fname == self._img_classes

        return x, y