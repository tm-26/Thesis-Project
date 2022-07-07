import csv
import datetime
import os
import shutil
import sys
import warnings
from distutils.dir_util import copy_tree
from predict import predict


if __name__ == "__main__":

    begin_time = datetime.datetime.now()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # Parameter Declaration
    dataset = "kdd17"
    retrainOnDrift = False
    retrainOnChange = True
    currentDirectory = os.getcwd()
    resultsFile = "../results/AdvALSTM - Method 5 - Sentiment Change.csv"
    # resultsFile = ''
    start = 40
    end = 41

    for i in range(start, end):

        numberOfWaitDays = i

        print("-----------------------------------------------------------------")
        print("Now doing numberOfWaitDays = " + str(numberOfWaitDays))
        print("-----------------------------------------------------------------")

        if retrainOnDrift:
            acc, mcc = predict(True, 5, numberOfWaitDays, 0)

        elif retrainOnChange:
            acc, mcc = predict(True, 5, numberOfWaitDays, 1)

        else:
            sys.path.append("Adv-ALSTM")
            from pred_lstm import AWLSTM

            if dataset == "stocknet":
                date = "2015-10-01"
                alp = 1
                bet = 0.01
                fixInit = 0
                seq = 5
                unit = 4
                eps = 0.05
            elif dataset == "kdd17":
                date = "2016-01-04"
                alp = 0.001
                bet = 0.05
                fixInit = 1
                seq = 15
                unit = 16
                eps = 0.001

            if os.path.exists("../models/advAlstmTemp-" + dataset):
                shutil.rmtree("../models/advAlstmTemp-" + dataset)

            os.mkdir("../models/advAlstmTemp-" + dataset)
            copy_tree("../models/advAlstm-" + dataset, "../models/advAlstmTemp-" + dataset)

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

            exit()

        if resultsFile != '':
            with open(resultsFile, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([numberOfWaitDays, acc, mcc])

        print(datetime.datetime.now() - begin_time)