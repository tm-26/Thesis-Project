import csv
import math
import numpy
import os
import pandas
import shutil
import sys
import skmultiflow.drift_detection
import statistics
from distutils.dir_util import copy_tree

sys.path.append("Adv-ALSTM")
from pred_lstm import AWLSTM


def klDivergence(list1, list2):
    return sum(list1[i] * math.log2(list1[i]/list2[i]) for i in range(len(list1)))


def splitAndSaveConcept(start, skipped, previousCutOffPoint, cutOffPoint, prepedData, allDates, stockName, conceptNumber, conceptSplitSize, file):
    conceptPart = 0
    start += skipped
    counter = 0
    for j in range(previousCutOffPoint + skipped, cutOffPoint):
        if counter == conceptSplitSize:
            if conceptPart < 10:
                conceptPartNumber = '0' + str(conceptPart)
            else:
                conceptPartNumber = str(conceptPart)
            if len(prepedData.iloc[start:start + counter]) != len(allDates[start:start + counter]):
                saveConcept(stockName[:-4] + str(conceptNumber) + " part " + conceptPartNumber, prepedData.iloc[start:start + counter], allDates[start + 1:start + counter], file)
            else:
                saveConcept(stockName[:-4] + str(conceptNumber) + " part " + conceptPartNumber, prepedData.iloc[start:start + counter], allDates[start:start + counter], file)
            conceptPart += 1
            start += counter
            counter = 0
        counter += 1

    # Handle incomplete concept

    if conceptPart < 10:
        conceptPartNumber = '0' + str(conceptPart)
    else:
        conceptPartNumber = str(conceptPart)

    if len(prepedData.iloc[start:start + counter]) != len(allDates[start:start + counter]):
        saveConcept(stockName[:-4] + str(conceptNumber) + " part " + conceptPartNumber, prepedData.iloc[start:start + counter], allDates[start + 1:start + counter], file)
    else:
        saveConcept(stockName[:-4] + str(conceptNumber) + " part " + conceptPartNumber, prepedData.iloc[start:cutOffPoint],
                allDates[start:cutOffPoint], file)
    return start + (cutOffPoint - start)


def printMe(text):
    sys.stdout = sys.__stdout__
    print(text)
    sys.stdout = open(os.devnull, 'w')


# Save concept to file
def saveConcept(name, concept, dates, file):

    if os.path.isdir(file + '/' + name):
        shutil.rmtree(file + '/' + name)
    os.mkdir(file + '/' + name)
    os.mkdir(file + '/' + name + '/' + name)
    concept.to_csv(file + '/' + name + "/" + name + '/' + name + ".csv", index=False, header=False)
    with open(file + '/' + name + "/trading_dates.csv", "w+", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        for date in dates:
            writer.writerow([date])


def predict(forcePreProcessing, method, conceptSplitSize=5):

    absPath = os.path.abspath('.')

    isMethod3 = False
    isMethod4 = False
    isMethod5 = False

    if method == 3:
        method = 2
        isMethod3 = True
    elif method == 4:
        method = 2
        isMethod4 = True
    elif method == 5:
        method = 2
        isMethod5 = True

    # Enter folder containing data
    os.chdir("../data/kdd17/Numerical")

    folder = "trainingPointsV" + str(method)

    # If data does not exist, create it
    if not os.path.isdir(folder):
        printMe(folder + " not found. Creating it...")
        forcePreProcessing = True
        os.mkdir(folder)

    if forcePreProcessing:
        # Clear contents of file
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.mkdir(folder)

        tradingDatesFile = list((csv.reader(open("trading_dates.csv", newline=''))))
        allDates = []
        testingDates = []
        isTesting = False
        stocks = os.listdir('./ourpped')

        # Get all dates for the testing set
        for date in tradingDatesFile:
            allDates.extend(date)
            if date[0] == "2016-01-04":
                isTesting = True
            if isTesting:
                testingDates.extend(date)

        del tradingDatesFile

        for stockName in stocks:
            skipped = 0
            conceptNumber = 0
            previousCutOffPoint = 0
            first = True
            data = pandas.read_csv("price_long_50/" + stockName, index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1]
            prepedData = pandas.read_csv("ourpped/" + stockName, header=None)

            # Detect concepts for different stock
            hddmA = skmultiflow.drift_detection.hddm_a.HDDM_A()
            conceptLength = 0
            start = 0
            for i in range(len(data)):
                # if current date in testing set
                if data.keys()[i].strftime("%Y-%m-%d") in testingDates:
                    hddmA.add_element(data[i])
                    conceptLength += 1
                    if hddmA.detected_change():
                        if conceptLength >= 63:
                            conceptLength = 0
                            cutOffPoint = i
                            if method == 1:
                                saveConcept(stockName[:-4] + str(conceptNumber),
                                            prepedData.iloc[previousCutOffPoint + skipped:cutOffPoint],
                                            allDates[previousCutOffPoint + skipped:cutOffPoint], folder)
                            elif method == 2:
                                start = splitAndSaveConcept(start, skipped, previousCutOffPoint, cutOffPoint, prepedData, allDates, stockName, conceptNumber, conceptSplitSize, folder)
                            skipped = 0
                            previousCutOffPoint = cutOffPoint
                            first = False
                            conceptNumber += 1
                elif first:
                    skipped += 1

            # Handle last concept
            if method == 1:
                saveConcept(stockName[:-4] + str(conceptNumber), prepedData.iloc[-(2518 - previousCutOffPoint):],
                            allDates[-(2518 - previousCutOffPoint):], folder)
            else:
                cutOffPoint = 2518
                splitAndSaveConcept(start, skipped, previousCutOffPoint, cutOffPoint, prepedData, allDates, stockName, conceptNumber, conceptSplitSize, folder)

    os.chdir(folder)

    if os.path.exists("../../../models/advAlstmTemp-KDD17") and os.path.isdir("../../../models/advAlstmTemp-KDD17"):
        shutil.rmtree("../../../models/advAlstmTemp-KDD17")

    os.mkdir("../../../models/advAlstmTemp-KDD17")
    copy_tree("../../../models/advAlstm-KDD17", "../../../models/advAlstmTemp-KDD17")


    # Create model
    pure_LSTM = AWLSTM(
        data_path=os.listdir('.')[0] + '/' + os.listdir('.')[0],
        model_path="../../../models/advAlstmTemp-KDD17/model",
        model_save_path="../../../models/advAlstmTemp-KDD17/model",
        parameters={
            "seq": 15,
            "unit": 16,
            "alp": 0.001,
            "bet": 0.05,
            "eps": 0.001,
            "lr": 0.01
        },
        steps=1,
        epochs=150, batch_size=1024, gpu=1,
        tra_date="2016-01-04", val_date="2016-01-04", tes_date="2016-01-04", att=1,
        hinge=1, fix_init=1, adv=0,
        reload=1
    )

    accList = []
    mccList = []
    stdAccList = []
    stdMccList = []
    currentAccList = []
    currentMccList = []

    seqChanged = False
    first = True

    # Method 4 variables
    conceptCounter = 0
    learn = []
    firstTest = True
    previousAcc = 0
    conceptsToRemember = []
    temp = []
    allConcepts = os.listdir("../trainingPointsV1")
    conceptDrift = False
    changingStock = False
    currentlyWaiting = True

    for i in range(len(allConcepts)):
        temp.append(allConcepts[i])
        if i+1 == len(allConcepts):
            conceptsToRemember.append(temp)
            break
        if allConcepts[i][:-1] != allConcepts[i+1][:-1]:
            conceptsToRemember.append(temp)
            temp = []

    currentConceptsToRemember = conceptsToRemember[0]
    del conceptsToRemember[0]
    del allConcepts

    listOfConcepts = os.listdir()
    for i in range(len(listOfConcepts)):

        # If temp, would have been handled already
        if listOfConcepts[i] == "temp":
            continue

        if seqChanged:
            pure_LSTM.set_seq(15)
            seqChanged = False

        dates = list((csv.reader(open(listOfConcepts[i] + "/trading_dates.csv"))))

        # If concept is to small, it would have already been handled with the previous concept
        if len(dates) <= 2:
            continue

        done = False

        # If next concept exists
        if i+1 != len(listOfConcepts):
            nextDates = list((csv.reader(open(listOfConcepts[i+1] + "/trading_dates.csv", newline=''))))
            # If next concept is to small (less then 2 dates), handle it now
            if len(nextDates) <= 2:
                current = pandas.concat([pandas.read_csv(listOfConcepts[i] + '/' + listOfConcepts[i] + '/' + listOfConcepts[i] + ".csv", header=None), pandas.read_csv(listOfConcepts[i+1] + '/' + listOfConcepts[i+1] + '/' + listOfConcepts[i+1] + ".csv",header=None)])
                with open(listOfConcepts[i] + "/trading_dates.csv", newline='') as f:
                    dates = [line.rstrip() for line in f]
                for date in nextDates:
                    dates.extend(date)
                saveConcept("temp", current, dates, '.')
                done = True
                pure_LSTM.set_data_path("temp/temp")
                startDate = dates[0]
                endDate = dates[-1]

                if isMethod4 or isMethod5:
                    if i + 2 != len(listOfConcepts) and listOfConcepts[i + 1].split(' ', 1)[0][:-1] != \
                            listOfConcepts[i + 2].split(' ', 1)[0][:-1]:
                        changingStock = True
                    elif i + 2 != len(listOfConcepts) and listOfConcepts[i + 1].split(' ', 1)[0] != \
                            listOfConcepts[i + 2].split(' ', 1)[0]:
                        conceptDrift = True

        # If dates are not set yet, set them
        if not done:
            pure_LSTM.set_data_path(listOfConcepts[i] + '/' + listOfConcepts[i])
            startDate = dates[0][0]
            endDate = dates[-1][0]

        # Test on current concept
        if len(dates) < 30:
            pure_LSTM.set_seq(math.floor(len(dates) / 2))
            seqChanged = True

        pure_LSTM.set_dates(startDate, startDate, startDate)

        result = pure_LSTM.test(True)
        if result is not None:
            currentAccList.append(result["acc"])
            currentMccList.append(result["mcc"])
        elif len(currentAccList) != 0:
            result = {"acc": currentAccList[-1], "mcc": currentMccList[-1]}
        elif len(accList) != 0:
            result = {"acc": accList[-1], "mcc": mccList[-1]}
        else:
            continue

        # Train Concept for future use

        # Check if current concept is complete (not last concept of stock)
        if i + 1 != len(listOfConcepts) and ((method == 1 and listOfConcepts[i][:-1] == listOfConcepts[i + 1][:-1]) or (
                method == 2 and listOfConcepts[i].split(' ', 1)[0] == listOfConcepts[i + 1].split(' ', 1)[0])):
            if first:
                pure_LSTM.set_model_path("../../../models/advAlstmTemp-KDD17/model")
                first = False
            pure_LSTM.set_dates(startDate, endDate, endDate)
            pure_LSTM.train(trainOnly=True)
        else:
            # Used for standard divergence calculation
            # accList.append(statistics.mean(currentAccList))
            # mccList.append(statistics.mean(currentMccList))

            if len(currentAccList) == 0 or len(currentMccList) == 0:
                continue

            accList.extend(currentAccList)
            mccList.extend(currentMccList)
            stdAccList.append(statistics.mean(currentAccList))
            stdMccList.append(statistics.mean(currentMccList))
            currentAccList = []
            currentMccList = []

        # Method 3
        if isMethod3 and i + 1 != len(listOfConcepts) and listOfConcepts[i].split(' ', 1)[0] != \
                listOfConcepts[i + 1].split(' ', 1)[0]:
            if os.path.exists("../../../models/advAlstmTemp-KDD17"):
                shutil.rmtree("../../../models/advAlstmTemp-KDD17")

            os.mkdir("../../../models/advAlstmTemp-KDD17")
            copy_tree("../../../models/advAlstm-KDD17", "../../../models/advAlstmTemp-KDD17")

            pure_LSTM.set_model_path("../../../models/advAlstmTemp-KDD17/model")

        # Method 4
        if isMethod4 or isMethod5:
            # If concept drift or changing stock
            if i + 1 != len(listOfConcepts) and listOfConcepts[i].split(' ', 1)[0] != listOfConcepts[i + 1].split(' ', 1)[0] or conceptDrift or changingStock:
                # re-init model
                if os.path.exists("../../../models/advAlstmTemp-KDD17"):
                    shutil.rmtree("../../../models/advAlstmTemp-KDD17")

                os.mkdir("../../../models/advAlstmTemp-KDD17")
                copy_tree("../../../models/advAlstm-KDD17", "../../../models/advAlstmTemp-KDD17")

                pure_LSTM.set_model_path("../../../models/advAlstmTemp-KDD17/model")

                firstTest = True
                currentlyWaiting = True

                # if changing stock
                if listOfConcepts[i].split(' ', 1)[0][:-1] != listOfConcepts[i + 1].split(' ', 1)[0][:-1] or changingStock:
                    # printMe("changing stock")
                    if changingStock:
                        changingStock = False
                    learn = []
                    conceptCounter = 0
                    currentConceptsToRemember = conceptsToRemember[0]
                    del conceptsToRemember[0]
                # if concept drift
                elif listOfConcepts[i].split(' ', 1)[0] != listOfConcepts[i + 1].split(' ', 1)[0] or conceptDrift:
                    if conceptDrift:
                        conceptDrift = False
                    conceptCounter += 1

                    # if concept drift has occurred twice or more, create list of concepts
                    if conceptCounter >= 2:
                        learn = list(range(0, conceptCounter - 1))

            # If there are concepts to learn
            if learn:
                if isMethod4:
                    if not currentlyWaiting:
                        printMe("learn list: " + str(learn))
                        # if first part of new concept, get an acc for comparison and train on a previous concept
                        if firstTest:
                            printMe("firstTest")
                            firstTest = False
                            previousAcc = result["acc"]
                            pure_LSTM.save_model("../../../models/advAlstmTemp-KDD17/model")
                            dates = list((csv.reader(open("../trainingPointsV1/" + currentConceptsToRemember[learn[0]] + "/trading_dates.csv"))))
                            pure_LSTM.set_data_path("../trainingPointsV1/" + currentConceptsToRemember[learn[0]] + '/' + currentConceptsToRemember[learn[0]])
                            pure_LSTM.set_dates(dates[0][0], dates[-1][0], dates[-1][0])
                            pure_LSTM.train(trainOnly=True)
                        else:
                            # if concept did not help, remove it
                            if previousAcc > result["acc"]:
                                printMe("forgetting " + currentConceptsToRemember[learn[0]])
                                pure_LSTM.set_model_path("../../../models/advAlstmTemp-KDD17/model")
                                firstTest = True
                            else:
                                printMe("keeping " + currentConceptsToRemember[learn[0]])
                                previousAcc = result["acc"]
                            del learn[0]
                    else:
                        currentlyWaiting = False
                elif isMethod5:
                    if len(learn) == 1:
                        dates = list((csv.reader(open("../trainingPointsV1/" + currentConceptsToRemember[learn[0]] + "/trading_dates.csv"))))
                        pure_LSTM.set_data_path("../trainingPointsV1/" + currentConceptsToRemember[learn[0]] + '/' + currentConceptsToRemember[learn[0]])
                        pure_LSTM.set_dates(dates[0][0], dates[-1][0], dates[-1][0])
                        pure_LSTM.train(trainOnly=True)
                    else:
                        divergences = []
                        dates = list((csv.reader(open("../trainingPointsV2/" + listOfConcepts[i] + "/trading_dates.csv"))))
                        currentDistribution = pandas.read_csv("../price_long_50/" + listOfConcepts[i].split(' ')[0][:-1] + ".csv",index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1]
                        currentDistribution = currentDistribution.iloc[numpy.where(currentDistribution.index == dates[0][0])[0][0]:(numpy.where(currentDistribution.index == dates[-1][0])[0][0]) + 1].values.tolist()
                        for j in range(len(learn)):
                            dates = list((csv.reader(open("../trainingPointsV1/" + currentConceptsToRemember[learn[j]] + "/trading_dates.csv"))))
                            data = pandas.read_csv("../price_long_50/" + currentConceptsToRemember[learn[j]][:-1] + ".csv",index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1]
                            divergences.append(klDivergence(data.iloc[numpy.where(data.index == dates[0][0])[0][0]:numpy.where(data.index == dates[len(currentDistribution)][0])[0][0]].values.tolist(), currentDistribution))
                        # Train model on most similar concept
                        printMe("divergences = " + str(divergences))
                        # selected = divergences.index(max(divergences))
                        selected = divergences.index(min(divergences, key=abs))
                        dates = list((csv.reader(open("../trainingPointsV1/" + currentConceptsToRemember[learn[selected]] + "/trading_dates.csv"))))
                        pure_LSTM.set_data_path("../trainingPointsV1/" + currentConceptsToRemember[learn[selected]] + '/' + currentConceptsToRemember[learn[selected]])
                        pure_LSTM.set_dates(dates[0][0], dates[-1][0], dates[-1][0])
                        pure_LSTM.train(trainOnly=True)
                    learn = []

        printMe("----" + listOfConcepts[i] + "----")
        printMe("ACC = " + str(result["acc"]))
        printMe("MCC = " + str(result["mcc"]))


    finalACC = sum(accList) / len(accList)
    finalMCC = sum(mccList) / len(mccList)

    printMe("Final ACC = " + str(finalACC))
    printMe("with standard deviation = " + str(statistics.stdev(stdAccList)))
    printMe("Final MCC = " + str(finalMCC))
    # printMe(stdMccList)
    printMe("with standard deviation = " + str(statistics.stdev(stdMccList)))

    os.chdir(absPath)
    sys.stdout = sys.__stdout__


    return str(finalACC * 100) + " ±" + str(statistics.stdev(accList)), str(finalMCC) + " ±" + str(statistics.stdev(mccList))


# if __name__ == "__main__":
#
#     # Handle arguments
#     argParser = argparse.ArgumentParser()
#     # argParser.add_argument("-m", "--model", help="Model that is going to be used: pure_lstm or att_lstm or adv_att_lstm", type=str, default="pure_lstm")
#     argParser.add_argument("-me", "--method", help="Method that is going to be used", type=int, default=2)
#     argParser.add_argument("-f", "--first", type=int, default=26)
#     argParser.add_argument("-l", "--last", type=int, default=26)
#
#     args = argParser.parse_args()
#
#     sys.stdout = open(os.devnull, 'w')  # Remove normal printing due to excess amount of printing in pred_lstm.py
#     absPath = os.path.abspath('.')
#     if 0 >= args.method <= 5:
#         printMe("Method needs to be a value between 1 and 4 not " + str(args.method))
#         exit(1)
#
#     if args.method == 1:
#         main(True, args.method)
#     else:
#         results = pandas.read_csv("AdvLstm with CDD Results (Method " + str(args.method) + ").csv")
#         complete = []
#         for i in results["Concept split size"]:
#             complete.append(int(i))
#
#         for i in range(args.first, args.last + 1):
#             if i not in complete:
#                 printMe("+------------------------------+")
#                 printMe("Doing method " + str(args.method) + ' (' + str(i) + ')')
#                 printMe("+------------------------------+")
#                 acc, mcc = main(True, args.method, i)
#                 os.chdir(absPath)
#                 results = results.append({"Concept split size": i, "ACC": acc, "MCC": mcc}, ignore_index=True)
#                 results.to_csv("AdvLstm with CDD Results (Method " + str(args.method) + ").csv", index=False)
#         results.plot(x="Concept split size", y="ACC")
#         results.plot(x="Concept split size", y="MCC")
#         matplotlib.pyplot.show()
