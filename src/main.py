import os
import shutil
import sys
from distutils.dir_util import copy_tree


if __name__ == "__main__":
    # Parameter Declaration
    dataset = "KDD17"
    method = 0

    if method == 0:
        if os.path.exists("../models/advAlstmTemp-" + dataset):
            shutil.rmtree("../models/advAlstmTemp-" + dataset)

        os.mkdir("../models/advAlstmTemp-" + dataset)
        copy_tree("../models/advAlstm-" + dataset, "../models/advAlstmTemp-" + dataset)

        os.chdir("Adv-ALSTM")

        if dataset == "KDD17":
            os.system("python pred_lstm.py -p ../../data/kdd17/Numerical/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -rl 1 -q ../../models/alstm-KDD17/alstm-KDD17 -la 0.05 -le 0.001 -f 1 -qs ../../models/advAlstmTemp-KDD17/advAlstm-KDD17")