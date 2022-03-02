import csv
import json
import os
import pandas
import pickle
import statistics
from decimal import Decimal, ROUND_HALF_UP
from finbert_embedding.embedding import FinbertEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch import flatten, reshape, stack
from torch.nn.functional import kl_div
from tqdm import tqdm


def createEmbeddings(dataset):
    finbertEmbedder = FinbertEmbedding()
    saveMe = []

    if dataset == "kdd17" or dataset == "stocknet":
        for stock in tqdm(os.listdir("../data/" + dataset + "/NYT-Business/ourpped")):
            with open("../data/" + dataset + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
                csvReader = csv.reader(file)
                next(csvReader)  # Skip headers
                for row in csvReader:
                    data = json.loads(row[1])
                    if type(data) is list:
                        for article in data:
                            saveMe.append([article["abstract"], finbertEmbedder.sentence_vector(article["abstract"])])
                    else:
                        saveMe.append([data["abstract"], finbertEmbedder.sentence_vector(data["abstract"])])
        with open("../data/" + dataset + "/NYT-Business/embeddings.pkl", "wb") as outputFile:
            pickle.dump(saveMe, outputFile)
    # elif dataset == "economynews":
    #     with open("../data/economyNews/economyNews.json", encoding="UTF-8") as file:
    #         economyNews = json.load(file)
    #         for point in economyNews:
    #             dataList.append([point["headlineText"], label])
    # elif dataset == "phrasebank":
    #     with open("../data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", encoding="ISO-8859-1") as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             dataList.append([line.rsplit(' ', 1)[0], label])
    # elif dataset == "bbcsport":
    #     for area in os.listdir("../data/bbcsport"):
    #         for document in os.listdir("../data/bbcsport/" + area):
    #             skip = False
    #             with open("../data/bbcsport/" + area + '/' + document) as file:
    #                 lines = file.readlines()
    #                 for line in lines:
    #                     if not skip:
    #                         dataList.append([line.replace('\n', ''), label])
    #                         skip = True
    #                     else:
    #                         skip = False
    # elif dataset == "slsamazon":
    #     with open("../data/SLS/amazon_cells_labelled.txt") as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             dataList.append([line[:-3], label])
    # elif dataset == "slsimbd":
    #     with open("../data/SLS/imdb_labelled.txt") as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             dataList.append([line[:-3], label])
    # elif dataset == "slsyelp":
    #     with open("../data/SLS/yelp_labelled.txt") as file:
    #         lines = file.readlines()
    #         for line in lines:
    #             dataList.append([line[:-3], label])
    # elif dataset == "stsgold":
    #     with open("../data/STS-Gold/sts_gold_tweet.csv") as file:
    #         reader = csv.reader(file)
    #         next(reader, None)  # Skip headers
    #         for row in reader:
    #             dataList.append([row[2], label])

    else:
        print("Parameter error: dataset=" + str(dataset) + " is not an accepted input")
        exit()


def getData(dataset, label):
    dataList = []

    if label == "embedding":
        if dataset == "kdd17" or dataset == "stocknet":
            with open("../data/" + dataset + "/NYT-Business/embeddings.pkl", "rb") as file:
                embeddings = pickle.load(file)
                for embedding in embeddings:
                    dataList.append(embedding[1])
            return dataList
    else:
        if dataset == "kdd17" or dataset == "stocknet":
            for stock in os.listdir("../data/" + dataset + "/NYT-Business/ourpped"):
                with open("../data/" + dataset + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
                    csvReader = csv.reader(file)
                    next(csvReader)  # Skip headers
                    for row in csvReader:
                        data = json.loads(row[1])
                        if type(data) is list:
                            for article in data:
                                dataList.append([article["abstract"], label])
                        else:
                            dataList.append([data["abstract"], label])
        elif dataset == "economynews":
            with open("../data/economyNews/economyNews.json", encoding="UTF-8") as file:
                economyNews = json.load(file)
                for point in economyNews:
                    dataList.append([point["headlineText"], label])
        elif dataset == "phrasebank":
            with open("../data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", encoding="ISO-8859-1") as file:
                lines = file.readlines()
                for line in lines:
                    dataList.append([line.rsplit(' ', 1)[0], label])
        elif dataset == "bbcsport":
            for area in os.listdir("../data/bbcsport"):
                for document in os.listdir("../data/bbcsport/" + area):
                    skip = False
                    with open("../data/bbcsport/" + area + '/' + document) as file:
                        lines = file.readlines()
                        for line in lines:
                            if not skip:
                                dataList.append([line.replace('\n', ''), label])
                                skip = True
                            else:
                                skip = False
        elif dataset == "slsamazon":
            with open("../data/SLS/amazon_cells_labelled.txt") as file:
                lines = file.readlines()
                for line in lines:
                    dataList.append([line[:-3], label])
        elif dataset == "slsimbd":
            with open("../data/SLS/imdb_labelled.txt") as file:
                lines = file.readlines()
                for line in lines:
                    dataList.append([line[:-3], label])
        elif dataset == "slsyelp":
            with open("../data/SLS/yelp_labelled.txt") as file:
                lines = file.readlines()
                for line in lines:
                    dataList.append([line[:-3], label])
        elif dataset == "stsgold":
            with open("../data/STS-Gold/sts_gold_tweet.csv") as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip headers
                for row in reader:
                    dataList.append([row[2], label])

        else:
            print("Parameter error: dataset=" + str(dataset) + " is not an accepted input")
            exit()

        return dataList


if __name__ == "__main__":

    # Parameter Declaration
    sourceDataset = "kdd17"  # Can be "kdd17" or "stocknet" or "economynews" or "phrasebank" or "bbcsport"
    # or "slsamazon" or "slsimbd" or "slsyelp" or "stsgold"
    targetDataset = "stocknet"  # Can be "kdd17" or "stocknet" or "economynews" or "phrasebank" or "bbcsport"
    # or "slsamazon" or "slsimbd" or "slsyelp" or "stsgold"
    distanceMetric = "kl"  # Can be either "pad" or "kl"
    numberOfIterations = 10
    showOtherMetrics = False  # Used in conjunction with distanceMetric=pad
    roundAnswers = True
    createEmbeddingsFiles = False
    
    # Variable Declaration
    iterationCounter = 0
    
    if createEmbeddingsFiles:
        createEmbeddings(sourceDataset)
        createEmbeddings(targetDataset)

    if distanceMetric == "pad":
        dataList = getData(sourceDataset, 0) + getData(targetDataset, 1)
        padList = []

        while iterationCounter < numberOfIterations:
            dataFrame = pandas.DataFrame(dataList, columns=["Text", "Origin"])
            xTrain, xTest, yTrain, yTest = train_test_split(
                TfidfVectorizer(strip_accents="unicode").fit_transform(dataFrame["Text"]), dataFrame["Origin"],
                test_size=0.2)
            svcModel = make_pipeline(StandardScaler(with_mean=False), LinearSVC(class_weight="balanced"))
            svcModel.fit(xTrain, yTrain)
            predicted = svcModel.predict(xTest)
            mae = mean_absolute_error(yTest, predicted)
            padList.append(2 * (1 - 2 * mae))

            if showOtherMetrics:
                print("MAE = " + str(mae) + ", F1 = " + str(f1_score(yTest, predicted)))
            print("(#" + str(iterationCounter + 1) + ") PAD from " + sourceDataset + " to " + targetDataset + " = " + str(
                padList[-1]))

            iterationCounter += 1
        if roundAnswers:
            print("Average PAD from " + sourceDataset + " to " + targetDataset + " = " + str(Decimal(
                Decimal(sum(padList) / len(padList)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))) + " (±" + str(
                Decimal(Decimal(statistics.stdev(padList)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))) + ')')
        else:
            print("Average PAD from " + sourceDataset + " to " + targetDataset + " = " + str(
                sum(padList) / len(padList)) + " ±" + str(statistics.stdev(padList)))

    elif distanceMetric == "kl":
        sourceData = stack(getData(sourceDataset, "embedding"))
        targetData = stack(getData(targetDataset, "embedding"))

        # reshape(targetData, list(sourceData.size()))
        flatten(targetData)
        flatten(sourceData)

        if roundAnswers:
            print("Kl-distance from " + sourceDataset + " to " + targetDataset + " = " + str(Decimal(
                Decimal(kl_div(sourceData, targetData, reduction="none").item()).quantize(Decimal("0.0001"),rounding=ROUND_HALF_UP))))
        else:
            print("Kl-distance from " + sourceDataset + " to " + targetDataset + " = " + str(kl_div(sourceData, targetData, reduction="batchmean").item()))

    else:
        print("Parameter error: distanceMetric=" + str(distanceMetric) + " is not an accepted input")
