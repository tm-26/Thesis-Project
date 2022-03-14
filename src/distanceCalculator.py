import csv
import json
import numpy
import os
import pandas
import pickle
import statistics
import torch
from decimal import Decimal, ROUND_HALF_UP
from finbert_embedding.embedding import FinbertEmbedding
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch.nn.functional import kl_div
from tqdm import tqdm


def equalizeDate(p, q):
    if p.shape[0] > q.shape[0]:
        q = torch.cat((q, torch.zeros([p.shape[0] - q.shape[0], 768])))
    elif p.shape[0] < q.shape[0]:
        p = torch.cat((p, torch.zeros([q.shape[0] - p.shape[0], 768])))

    return p, q


def calculateDistance(sourceDataset, targetDataset):
    iterationCounter = 0

    if createEmbeddingsFiles:
        # createEmbeddings(sourceDataset)
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
            print(
                "(#" + str(iterationCounter + 1) + ") PAD from " + sourceDataset + " to " + targetDataset + " = " + str(
                    padList[-1]))

            iterationCounter += 1
        if roundAnswers:
            print("Average PAD from " + sourceDataset + " to " + targetDataset + " = " + str(Decimal(
                Decimal(sum(padList) / len(padList)).quantize(Decimal("0.0001"),
                                                              rounding=ROUND_HALF_UP))) + " (±" + str(
                Decimal(Decimal(statistics.stdev(padList)).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))) + ')')
        else:
            print("Average PAD from " + sourceDataset + " to " + targetDataset + " = " + str(
                sum(padList) / len(padList)) + " ±" + str(statistics.stdev(padList)))

    elif distanceMetric == "kl":
        """
        N.B, KL Distance has the limitation that the two lists should be equal. If they are not the longer one 
        will be shortened to the length of the shorter one. 
        """

        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")),
                                              torch.stack(getData(targetDataset, "embedding")))

        if roundAnswers:
            print("Kl distance from " + sourceDataset + " to " + targetDataset + " = " + str(Decimal(
                Decimal(kl_div(sourceData, targetData, reduction="batchmean").item()).quantize(Decimal("0.0001"),
                                                                                               rounding=ROUND_HALF_UP))))
        else:
            print("Kl distance from " + sourceDataset + " to " + targetDataset + " = " + str(
                kl_div(sourceData, targetData, reduction="batchmean").item()))

    elif distanceMetric == "hellinger":
        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")),
                                              torch.stack(getData(targetDataset, "embedding")))

        if roundAnswers:
            print("Hellinger distance from " + sourceDataset + " to " + targetDataset + " = " + str(Decimal(Decimal(
                numpy.sqrt(numpy.nansum((numpy.sqrt(sourceData) - numpy.sqrt(targetData)) ** 2)) / numpy.sqrt(
                    2)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))))
        else:
            print("Hellinger distance from " + sourceDataset + " to " + targetDataset + " = " + str(
                numpy.sqrt(numpy.nansum((numpy.sqrt(sourceData) - numpy.sqrt(targetData)) ** 2)) / numpy.sqrt(2)))

    elif distanceMetric == "tv":

        sourceData, targetData = equalizeDate(torch.stack(getData(sourceDataset, "embedding")),
                                              torch.stack(getData(targetDataset, "embedding")))

        if roundAnswers:
            print("Total variation distance from " + sourceDataset + " to " + targetDataset + " = " +
                  str(Decimal(Decimal(numpy.float(numpy.nansum(numpy.absolute(sourceData.numpy() - targetData.numpy()))))).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)))
        else:
            print("Total variation distance from " + sourceDataset + " to " + targetDataset + " = " + str(
                numpy.sqrt(numpy.nansum((numpy.sqrt(sourceData) - numpy.sqrt(targetData)) ** 2)) / numpy.sqrt(2)))

    else:
        print("Parameter error: distanceMetric=" + str(distanceMetric) + " is not an accepted input")


def getEmbeddings(file):
    dataList = []

    embeddings = pickle.load(file)
    for embedding in embeddings:
        dataList.append(embedding[1])
    return dataList


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
        pickle.dump(saveMe, open("../data/" + dataset + "/NYT-Business/embeddings.pkl", "wb"))
    elif dataset == "economynews":
        with open("../data/economyNews/economyNews.json", encoding="UTF-8") as file:
            economyNews = json.load(file)
            for point in tqdm(economyNews):
                saveMe.append([point["headlineText"], finbertEmbedder.sentence_vector(point["headlineText"])])

        pickle.dump(saveMe, open("../data/economyNews/embeddings.pkl", "wb"))
    elif dataset == "phrasebank":
        with open("../data/FinancialPhraseBank-v1.0/Sentences_50Agree.txt", encoding="ISO-8859-1") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line.rsplit(' ', 1)[0]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])

        pickle.dump(saveMe, open("../data/FinancialPhraseBank-v1.0/embeddings.pkl", "wb"))
    elif dataset == "bbcsport":
        for area in tqdm(["athletics", "cricket", "football", "rugby", "tennis"]):
            for document in os.listdir("../data/bbcsport/" + area):
                skip = False
                with open("../data/bbcsport/" + area + '/' + document) as file:
                    lines = file.readlines()
                    for line in lines:
                        if not skip:
                            line = line.replace('\n', '')
                            saveMe.append([line, finbertEmbedder.sentence_vector(line)])
                            skip = True
                        else:
                            skip = False
        pickle.dump(saveMe, open("../data/bbcsport/embeddings.pkl", "wb"))
    elif dataset == "slsamazon":
        with open("../data/SLS/amazon_cells_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/amazonEmbeddings.pkl", "wb"))
    elif dataset == "slsimbd":
        with open("../data/SLS/imdb_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/imbdEmbeddings.pkl", "wb"))
    elif dataset == "slsyelp":
        with open("../data/SLS/yelp_labelled.txt") as file:
            lines = file.readlines()
            for line in tqdm(lines):
                line = line[:-3]
                saveMe.append([line, finbertEmbedder.sentence_vector(line)])
        pickle.dump(saveMe, open("../data/SLS/yelpEmbeddings.pkl", "wb"))
    elif dataset == "stsgold":
        with open("../data/STS-Gold/sts_gold_tweet.csv") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip headers
            for row in tqdm(reader):
                saveMe.append([row[2], finbertEmbedder.sentence_vector(row[2])])
        pickle.dump(saveMe, open("../data/STS-Gold/embeddings.pkl", "wb"))

    else:
        print("Parameter error: dataset=" + str(dataset) + " is not an accepted input")
        exit()


def getData(dataset, label):
    if label == "embedding":
        if dataset == "kdd17" or dataset == "stocknet":
            return getEmbeddings(open("../data/" + dataset + "/NYT-Business/embeddings.pkl", "rb"))
        elif dataset == "economynews":
            return getEmbeddings(open("../data/economyNews/embeddings.pkl", "rb"))
        elif dataset == "phrasebank":
            return getEmbeddings(open("../data/FinancialPhraseBank-v1.0/embeddings.pkl", "rb"))
        elif dataset == "bbcsport":
            return getEmbeddings(open("../data/bbcsport/embeddings.pkl", "rb"))
        elif dataset == "slsamazon":
            return getEmbeddings(open("../data/SLS/amazonEmbeddings.pkl", "rb"))
        elif dataset == "slsimbd":
            return getEmbeddings(open("../data/SLS/imbdEmbeddings.pkl", "rb"))
        elif dataset == "slsyelp":
            return getEmbeddings(open("../data/SLS/yelpEmbeddings.pkl", "rb"))
        elif dataset == "stsgold":
            return getEmbeddings(open("../data/STS-Gold/embeddings.pkl", "rb"))
    else:
        dataList = []
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
    sourceDataset = "stsgold"  # Can be "kdd17" or "stocknet" or "economynews" or "phrasebank" or "bbcsport"
    # or "slsamazon" or "slsimbd" or "slsyelp" or "stsgold"
    targetDataset = "all"  # Can be "kdd17" or "stocknet" or "economynews" or "phrasebank" or "bbcsport"
    # or "slsamazon" or "slsimbd" or "slsyelp" or "stsgold" or "all"
    distanceMetric = "tv"  # Can be either "pad" or "kl" or "hellinger" or "tv"
    numberOfIterations = 10
    showOtherMetrics = False  # Used in conjunction with distanceMetric=pad
    roundAnswers = True
    createEmbeddingsFiles = False

    if targetDataset == "all":
        for dataset in ["kdd17", "stocknet", "economynews", "phrasebank", "bbcsport", "slsamazon", "slsimbd", "slsyelp",
                        "stsgold"]:
            calculateDistance(sourceDataset, dataset)
    else:
        calculateDistance(sourceDataset, targetDataset)
