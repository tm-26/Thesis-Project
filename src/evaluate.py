import csv
import json
import logging
import matplotlib.pyplot
import os
import pandas
import pickle
import random
import sklearn
from finbert import predict
from finbert_embedding.embedding import FinbertEmbedding
from statistics import mean
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from torch.nn.functional import kl_div


def getDistances(data):
    values = {}

    for point in tqdm(data):
        prediction = True
        distance = round(
            kl_div(chosenEmbedding, finbertEmbedder.sentence_vector(point[0]), reduction="batchmean").item(), 4)

        if point[1] != point[2]:
            prediction = False

        if distance in values:
            values[distance].append(prediction)
        else:
            values[distance] = [prediction]

    return values


def generateAccuracyFile():
    finBert = AutoModelForSequenceClassification.from_pretrained("../models/FinBert", cache_dir=None, num_labels=3)
    points = []
    predictions = []
    groundTruths = []

    with open("../data/economyNews/economyNews.json", encoding="utf-8") as file:
        data = json.load(file)
        for point in tqdm(data):
            scores = []
            groundTruths.append(point["classification"])

            answer = predict(point["headlineText"], finBert)
            for j, score in enumerate(answer["sentiment_score"]):
                scores.append(score)

            answer = predict(point["headlineTitle"], finBert)
            for j, score in enumerate(answer["sentiment_score"]):
                scores.append(score)

            if mean(scores) > 0:
                points.append((point["headlineText"] + ' ' + point["headlineTitle"], 1, point["classification"]))
                predictions.append(1)
            else:
                points.append((point["headlineText"] + ' ' + point["headlineTitle"], -1, point["classification"]))
                predictions.append(-1)

    with open("../results/finBert-EconomyNews.csv", "w+", encoding="UTF-8", newline='') as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(("Text", "Prediction", "Ground Truth"))
        for point in points:
            csvWriter.writerow(point)

    print("Calculated Accuracy = " + str(sklearn.metrics.accuracy_score(groundTruths, predictions)))


if __name__ == "__main__":

    logging.disable()

    # Parameter Declaration
    estimateScore = False  # True --> Estimate performance of FinBert, False --> Calculate real performance of FinBert
    generateFile = False
    newResultsFile = False
    showGraphs = True
    iterations = 0  # iterations = 0 -->  Calculate Metrics based on current values

    if generateFile:
        generateAccuracyFile()

    if estimateScore:
        iterationCounter = 0
        realAccuracies = []
        estimatedAccuracies = []

        with open("../results/finBert-EconomyNews.csv", "r", encoding="UTF-8") as file:
            csvReader = csv.reader(file)
            next(csvReader)  # skip header
            data = list(map(tuple, csvReader))

        if not os.path.isfile("../results/finBert-EconomyNews-EstimatedAccuracy.csv"):
            newResultsFile = True
        elif newResultsFile:
            os.remove("../results/finBert-EconomyNews-EstimatedAccuracy")

        with open("../results/finBert-EconomyNews-EstimatedAccuracy.csv", "a+", encoding="UTF-8",
                  newline="") as resultsFile:

            if newResultsFile:
                resultsFile.write("Estimated Accuracy,Real Accuracy\n")

            while iterationCounter < iterations:

                print("Iteration #" + str(iterationCounter + 1))

                random.shuffle(data)
                trainData = data[:int((len(data) + 1) * .80)]
                testData = data[int((len(data) + 1) * .80):]

                finbertEmbedder = FinbertEmbedding()
                chosenEmbedding = 0

                for point in trainData:
                    if point[1] == point[2]:
                        chosenEmbedding = finbertEmbedder.sentence_vector(point[0])
                        break

                trainValues = getDistances(trainData)
                testValues = getDistances(testData)

                dataFrameTrain = pandas.DataFrame(columns=["Domain-shift Detection Metric", "Classification Drop"])
                dataFrameTest = pandas.DataFrame(columns=["Domain-shift Detection Metric", "Classification Drop"])

                for i, (x, y) in enumerate(trainValues.items()):
                    dataFrameTrain.loc[i] = [x, len([prediction for prediction in y if not prediction]) / len(y)]

                for i, (x, y) in enumerate(testValues.items()):
                    dataFrameTest.loc[i] = [x, len([prediction for prediction in y if not prediction]) / len(y)]

                xTrain = dataFrameTrain.iloc[:, :-1].values
                xTest = dataFrameTest.iloc[:, :-1].values
                yTrain = dataFrameTrain.iloc[:, 1].values
                yTest = dataFrameTest.iloc[:, 1].values

                # xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(dataFrame.iloc[:, :-1].values, dataFrame.iloc[:, 1].values, test_size=0.2, random_state=0)
                model = sklearn.linear_model.LinearRegression()
                model.fit(xTrain, yTrain)

                pickle.dump(model, open("../models/accuracyPredictor.sav", "wb+"))

                yPredicted = model.predict(xTest)

                if showGraphs:
                    matplotlib.pyplot.rcParams["figure.autolayout"] = True
                    matplotlib.pyplot.scatter(xTest, yTest)
                    matplotlib.pyplot.plot(xTest, yPredicted, color="black", linewidth=3)

                    matplotlib.pyplot.xticks()
                    matplotlib.pyplot.yticks()

                    matplotlib.pyplot.show()

                pseudoTruths = []

                for distance in xTest:
                    prediction = model.predict([distance])[0]
                    if prediction == 0:
                        pseudoTruths.append(True)
                    elif prediction == 1:
                        pseudoTruths.append(False)
                    elif random.random() <= prediction:
                        pseudoTruths.append(False)
                    else:
                        pseudoTruths.append(True)

                estimatedAccuracy = len([pseudoTruth for pseudoTruth in pseudoTruths if pseudoTruth]) / len(pseudoTruths)

                print("Estimated Accuracy = " + str(estimatedAccuracy))

                estimatedAccuracies.append(estimatedAccuracy)

                realPredictions = []
                for point in testData:
                    if point[1] == point[2]:
                        realPredictions.append(True)
                    else:
                        realPredictions.append(False)

                realAccuracy = len([realPrediction for realPrediction in realPredictions if realPrediction]) / len(
                    realPredictions)

                print("Real Accuracy = " + str(realAccuracy))

                realAccuracies.append(realAccuracy)

                resultsFile.write(str(estimatedAccuracy) + "," + str(realAccuracy) + '\n')
                iterationCounter += 1

        results = pandas.read_csv("../results/finBert-EconomyNews-EstimatedAccuracy.csv")
        print("MAE = " + str(sklearn.metrics.mean_absolute_error(results["Real Accuracy"], results["Estimated Accuracy"])))
        print("MAX = " + str(abs(results["Estimated Accuracy"] - results["Real Accuracy"]).max()))

    else:
        results = pandas.read_csv("../results/finBert-EconomyNews.csv")
        print("ACC = " + str(sklearn.metrics.accuracy_score(results["Ground Truth"], results["Prediction"])))
        print("Precision = " + str(sklearn.metrics.precision_score(results["Ground Truth"], results["Prediction"])))
        print("Recall = " + str(sklearn.metrics.recall_score(results["Ground Truth"], results["Prediction"])))
        print("F1 = " + str(sklearn.metrics.f1_score(results["Ground Truth"], results["Prediction"])))