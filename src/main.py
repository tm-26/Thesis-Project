import datetime
import os
import shutil
import sys
from distutils.dir_util import copy_tree

sys.path.append("Adv-ALSTM")
from pred_lstm import AWLSTM

if __name__ == "__main__":

    begin_time = datetime.datetime.now()

    # Parameter Declaration
    dataset = "stocknet"
    method = 0

    # Default model parameters belong to KDD17 dataset
    date = "2016-01-04"
    alp = 0.001
    bet = 0.05
    fixInit = 1
    seq = 15
    unit = 16
    eps = 0.001

    if method == 0:
        if os.path.exists("../models/advAlstmTemp-" + dataset):
            shutil.rmtree("../models/advAlstmTemp-" + dataset)

        os.mkdir("../models/advAlstmTemp-" + dataset)
        copy_tree("../models/advAlstm-" + dataset, "../models/advAlstmTemp-" + dataset)

        if dataset == "stocknet":
            date = "2015-10-01"
            alp = 1
            bet = 0.01
            fixInit = 0
            seq = 5
            unit = 4
            eps = 0.05

        pure_LSTM = AWLSTM(
            data_path="../data/" + dataset + "/Numerical/ourpped/",
            model_path="../models/advAlstmTemp-" + dataset + "/model",
            # model_path="../models/tempModel/tempModel",
            model_save_path="../models/advAlstmTemp-" + dataset + "/model",
            parameters={
                "seq": seq,
                "unit": unit,
                "alp": alp,
                "bet": bet,
                "eps": eps,
                "lr": 0.01
            },
            steps=1,
            epochs=150, batch_size=1024, gpu=1,
            tra_date=date, val_date=date, tes_date=date, att=1,
            hinge=1, fix_init=fixInit, adv=1,
            reload=1
        )

        pure_LSTM.test(True)

        print(datetime.datetime.now() - begin_time)