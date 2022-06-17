import csv
import json
import matplotlib.pyplot
import os
import pandas
import sklearn
import skmultiflow.drift_detection
import sys
from datetime import timedelta
from detector import conceptDriftDetector
from detecta import detect_cusum
from statistics import mean
from tqdm import tqdm


def calculateCorrelationFunction(numberOfStocks, CDD, predictCD, considerMovement):
    # Variable Declaration
    dates = []
    stockPrices = []
    dailySentiment = []
    conceptDriftPoints = []
    sentimentChangePoints = []
    sentimentChangeDates = []
    labels = []
    numerical = []
    sentiment = []
    numberOfArticles = {}

    for counter, stockCode in enumerate(os.listdir("../data/kdd17/SentimentScores/NYT-Business/")):
        with open("../data/kdd17/SentimentScores/NYT-Business/" + stockCode, encoding="UTF-8") as file:
            numberOfArticles[stockCode[:-4]] = sum(1 for _ in file)

    for stockCode in tqdm(sorted(numberOfArticles, key=numberOfArticles.get, reverse=True)[:numberOfStocks]):
        # Getting numerical first
        currentNumerical = pandas.read_csv("../data/kdd17/Numerical/price_long_50/" + stockCode + ".csv", header=0).loc[
                           ::-1]
        currentNumerical["Date"] = pandas.to_datetime(currentNumerical["Date"])

        numerical.append(currentNumerical)

        # Getting sentiment next
        currentSentiment = pandas.read_csv("../data/kdd17/SentimentScores/NYT-Business/" + stockCode + ".csv", header=0)
        currentSentiment["Date"] = pandas.to_datetime(currentSentiment["Date"])

        sentiment.append(currentSentiment)

    numerical = pandas.concat(numerical)
    sentiment = pandas.concat(sentiment)

    sentimentScores = {}
    for index, row in sentiment.iterrows():
        if row["Date"] in sentimentScores:
            sentimentScores[row["Date"]].append(row["sentimentScore"])
        else:
            sentimentScores[row["Date"]] = [row["sentimentScore"]]

    startDate = pandas.Timestamp(numerical["Date"].iloc[0]) - pandas.DateOffset(days=1)
    endDate = pandas.Timestamp(numerical["Date"].iloc[-1]) + pandas.DateOffset(days=1)

    delta = endDate - startDate

    for counter in range(delta.days + 1):
        date = startDate + timedelta(days=counter)
        if date in sentimentScores:
            currentSentiment = mean(sentimentScores[date])
        else:
            currentSentiment = 0

        if (date == numerical["Date"]).any():
            stockPrices.append(numerical.loc[numerical["Date"] == date, "Adj Close"].iloc[0])
            dailySentiment.append(currentSentiment)
            dates.append(date)

    driftPoints = conceptDriftDetector(stockPrices, CDD, 0.00001)

    for i in range(len(dates)):
        if i in driftPoints:
            conceptDriftPoints.append(1)
        else:
            conceptDriftPoints.append(0)

    (sentimentDates, sentimentScores) = zip(*sentimentScores.items())
    changePoints = detect_cusum(list(map(lambda idx: sum(idx) / float(len(idx)), sentimentScores)), show=False)[
        0].tolist()

    for point in changePoints:
        # Needs to be done to account for days when stock market is not open
        date = sentimentDates[point]
        while True:
            if (date == numerical["Date"]).any():
                sentimentChangeDates.append(sentimentDates[point])
                break
            date = date + pandas.DateOffset(days=1)

    for date in dates:
        if date in sentimentChangeDates:
            sentimentChangePoints.append(1)
        else:
            sentimentChangePoints.append(0)

    # This fails if a drift happens in the first 5 days - Should not be the case, but worth pointing out nonetheless

    if predictCD:
        first = sentimentChangePoints
        second = conceptDriftPoints
    else:
        first = conceptDriftPoints
        second = sentimentChangePoints

    if considerMovement:
        for counter, drift in enumerate(first):
            if drift == 1 and 1 in second[counter: counter + 6]:
                index = counter + second[counter: counter + 6].index(1)
                if stockPrices[index] < stockPrices[counter]:
                    print(2)
                    labels.append(2)
                else:
                    labels.append(1)
            else:
                labels.append(0)
    else:
        for counter, drift in enumerate(first):
            if drift == 1 and 1 in second[counter: counter + 6]:
                labels.append(1)
            else:
                labels.append(0)

    x = pandas.DataFrame(
        {"Stock Price": stockPrices, "Sentiment Score": dailySentiment, "Concept Drift": conceptDriftPoints,
         "Sentiment Change": sentimentChangePoints})
    y = pandas.DataFrame({"Label": labels})

    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    model = sklearn.svm.SVC(kernel="rbf", gamma=0.5, class_weight="balanced")
    model.fit(xTrain, yTrain)
    predicted = model.predict(xTest)

    if considerMovement:
        return [sklearn.metrics.accuracy_score(yTest, predicted)]

    return [sklearn.metrics.accuracy_score(yTest, predicted), sklearn.metrics.f1_score(yTest, predicted),
            sklearn.metrics.confusion_matrix(yTest, predicted)]


def plotter(stockCode, saveLocation, start, end, threshold, generateDriftCircles, CDD, SCD, sentimentChange,
            thresholdSentiment, windowSize, showGraph, saveSentimentLocation, sentimentWindowSize, sentimentThreshold,
            volumeThreshold):
    # Getting numerical first
    numerical = pandas.read_csv("../data/kdd17/Numerical/price_long_50/" + stockCode + ".csv", header=0).loc[::-1]
    numerical["Date"] = pandas.to_datetime(numerical["Date"])
    numerical = numerical.loc[(numerical["Date"] > start) & (numerical["Date"] <= end)]

    if windowSize > 1:
        numerical["Adj Close"] = numerical["Adj Close"].rolling(windowSize).mean()

    # Getting sentiment next
    sentiment = pandas.read_csv("../data/kdd17/SentimentScores/NYT-Business/" + stockCode + ".csv", header=0)
    sentiment["Date"] = pandas.to_datetime(sentiment["Date"])
    sentiment = sentiment.loc[(sentiment["Date"] > start) & (sentiment["Date"] <= end)]

    sentimentScores = {}
    for index, row in sentiment.iterrows():
        if row["Date"] in sentimentScores:
            sentimentScores[row["Date"]].append(row["sentimentScore"])
        else:
            sentimentScores[row["Date"]] = [row["sentimentScore"]]

    dailySentiment = []
    startDate = pandas.Timestamp(start) - pandas.DateOffset(days=1)
    endDate = pandas.Timestamp(end) + pandas.DateOffset(days=1)
    allDates = []
    delta = endDate - startDate

    for counter in range(delta.days + 1):
        date = startDate + timedelta(days=counter)
        allDates.append(date)
        if date in sentimentScores:
            dailySentiment.append(mean(sentimentScores[date]))
        else:
            dailySentiment.append(0)

    # Plotting
    figure, axes = matplotlib.pyplot.subplots()
    axes.plot_date(numerical["Date"], numerical["Adj Close"], '-')
    axes.tick_params(axis='x', labelsize=9)

    sentimentFigure, sentimentAxes = matplotlib.pyplot.subplots()
    sentimentAxes.plot_date(allDates, dailySentiment, '-')
    sentimentAxes.tick_params(axis='x', labelsize=9)

    circleFormat = ""
    driftPoints = conceptDriftDetector(numerical["Adj Close"].tolist()[windowSize:], CDD, 0.00001)
    sentimentDriftDetector = None
    if SCD == "eddm":
        sentimentDriftDetector = skmultiflow.drift_detection.eddm.EDDM()
        previous = 0
    elif SCD == "hddma":
        sentimentDriftDetector = skmultiflow.drift_detection.hddm_a.HDDM_A(0.00001)
    elif SCD == "hddmw":
        sentimentDriftDetector = skmultiflow.drift_detection.hddm_w.HDDM_W()
    elif SCD == "ph":
        sentimentDriftDetector = skmultiflow.drift_detection.page_hinkley.PageHinkley()
    elif SCD == "adwin":
        sentimentDriftDetector = skmultiflow.drift_detection.adwin.ADWIN()
    elif SCD == "cusum":

        scores = list(map(lambda idx: sum(idx) / float(len(idx)), sentimentScores.values()))
        if sentimentWindowSize > 1:
            scores = pandas.Series(scores)
            scores = scores.rolling(sentimentWindowSize).mean().tolist()[sentimentWindowSize:]

        sentimentChangePoints = detect_cusum(scores, sentimentThreshold, show=False)[0].tolist()

    for i in range(len(driftPoints)):
        driftPoints[i] = numerical["Date"][driftPoints[i]]

    if volumeThreshold > 0:
        with open("../data/kdd17/NYT-Business/ourpped/" + stockCode + ".csv", encoding="UTF-8") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            allArticles = {row[0]: row[1] for row in reader}

    # For statistical purposes
    exact = 0
    before = {-1: 0, -2: 0, -3: 0, -4: 0, -5: 0, -6: 0, -7: 0, -8: 0, -9: 0, -10: 0}
    after = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    inline = 0
    numOfSentimentCircles = 0

    for index, date in enumerate(list(sentimentScores.keys())):

        current = mean(sentimentScores[date])

        if thresholdSentiment:

            if -threshold >= current:
                circleFormat = "or"
            elif threshold <= current:
                circleFormat = "og"
            else:
                continue

        elif sentimentChange:

            if SCD == "eddm":
                if previous < current:
                    sentimentDriftDetector.add_element(1)
                else:
                    sentimentDriftDetector.add_element(0)

                previous = current

                if sentimentDriftDetector.detected_change():
                    if current <= 0:
                        circleFormat = "or"
                    else:
                        circleFormat = "og"
                else:
                    continue

            elif SCD == "cusum":
                if len(sentimentChangePoints) == 0:
                    break

                if index == sentimentChangePoints[0]:
                    del sentimentChangePoints[0]

                    if current <= 0:
                        circleFormat = "or"
                    else:
                        circleFormat = "og"

                else:
                    continue

            else:
                sentimentDriftDetector.add_element(current)
                if sentimentDriftDetector.detected_change():
                    if current <= 0:
                        circleFormat = "or"
                    else:
                        circleFormat = "og"

                else:
                    continue

        if volumeThreshold > 0:
            currentArticles = allArticles[str(date)[:-9]]
            if currentArticles[0] != '[' or len(json.loads(currentArticles)) < volumeThreshold:
                continue

        numOfSentimentCircles += 1

        sentimentAxes.plot_date(date, current, circleFormat, fillstyle="none", ms=5.0)

        # Needs to be done to account for days when stock market is not open
        while True:
            try:
                axes.plot_date(date, numerical.loc[numerical["Date"] == date, "Adj Close"].iloc[0], circleFormat,
                               fillstyle="none", ms=5.0)
            except IndexError:
                date = date + pandas.DateOffset(days=1)
            else:
                break

        # Sort by dates closest to current
        driftPoints.sort(key=lambda i: abs(i - date))
        incremented = False
        for driftPoint in driftPoints:
            dateDifference = (driftPoint - date).days

            if abs(dateDifference) > 10:
                break

            if not incremented:
                incremented = True
                inline += 1

            if dateDifference == 0:
                exact += 1
            elif dateDifference < 0:
                before[dateDifference] += 1
            else:
                after[dateDifference] += 1

    if generateDriftCircles:
        for driftPoint in driftPoints:
            axes.plot_date(driftPoint, numerical.loc[numerical["Date"] == driftPoint, "Adj Close"].iloc[0], 'o',
                           fillstyle="none", ms=5.0, color="black")

    axes.title.set_text(stockCode)
    axes.set_xlabel("Time")
    axes.set_ylabel("Adj Close Price ($)")

    sentimentAxes.title.set_text(stockCode)
    sentimentAxes.set_xlabel("Time")
    # sentimentAxes.xaxis.set_major_locator(YearLocator())
    sentimentAxes.set_ylabel("Sentiment")

    if saveLocation != "":
        matplotlib.pyplot.figure(figure.number)
        matplotlib.pyplot.savefig(saveLocation)

    if saveSentimentLocation != "":
        matplotlib.pyplot.figure(sentimentFigure.number)
        matplotlib.pyplot.savefig(saveSentimentLocation)

    if showGraph:
        matplotlib.pyplot.show()

    else:

        figure.clear()
        matplotlib.pyplot.close(figure)
        sentimentFigure.clear()
        matplotlib.pyplot.close(sentimentFigure)

    return before, exact, after, len(driftPoints), inline, numOfSentimentCircles


def plot(stockCode="all", saveLocation='', start="2007-01-01", end="2017-01-01", threshold=0.8,
         generateDriftCircles=True, saveTextLocation='', CDD="hddma", SCD="cusum", sentimentChange=False,
         thresholdSentiment=False, windowSize=0, showGraph=False, saveSentimentLocation="", printValues=False,
         sentimentWindowSize=0, sentimentThreshold=1, volumeThreshold=0):
    # Variable Declaration
    totalBeforeDict = {-1: 0, -2: 0, -3: 0, -4: 0, -5: 0}
    totalExact = 0
    totalAfterDict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    totalNumDriftPoints = 0
    totalInline = 0
    totalNumOfSentimentCircles = 0

    if stockCode == "all":
        for stockCode in tqdm(os.listdir("../data/kdd17/Numerical/ourpped")):
            before, exact, after, totalDriftPoints, inline, numOfSentimentCircles = plotter(stockCode[-4], saveLocation,
                                                                                            start, end, threshold,
                                                                                            generateDriftCircles, CDD,
                                                                                            SCD, sentimentChange,
                                                                                            thresholdSentiment,
                                                                                            windowSize, showGraph,
                                                                                            saveSentimentLocation,
                                                                                            sentimentWindowSize,
                                                                                            sentimentThreshold,
                                                                                            volumeThreshold)

            for i in range(1, 6):
                totalBeforeDict[-i] += before[-i]
                totalAfterDict[i] += after[i]

            totalExact += exact
            totalNumDriftPoints += totalDriftPoints
            totalInline += inline
            totalNumOfSentimentCircles += numOfSentimentCircles
    elif isinstance(stockCode, int):

        numberOfArticles = {}
        numberOfStocks = stockCode

        for counter, stockCode in enumerate(os.listdir("../data/kdd17/SentimentScores/NYT-Business/")):
            with open("../data/kdd17/SentimentScores/NYT-Business/" + stockCode, encoding="UTF-8") as file:
                numberOfArticles[stockCode[:-4]] = sum(1 for _ in file)

        for stockCode in tqdm(sorted(numberOfArticles, key=numberOfArticles.get, reverse=True)[:numberOfStocks]):

            before, exact, after, totalDriftPoints, inline, numOfSentimentCircles = plotter(stockCode, saveLocation,
                                                                                            start, end, threshold,
                                                                                            generateDriftCircles, CDD,
                                                                                            SCD, sentimentChange,
                                                                                            thresholdSentiment,
                                                                                            windowSize, showGraph,
                                                                                            saveSentimentLocation,
                                                                                            sentimentWindowSize,
                                                                                            sentimentThreshold,
                                                                                            volumeThreshold)
            for i in range(1, 6):
                totalBeforeDict[-i] += before[-i]
                totalAfterDict[i] += after[i]

            totalExact += exact
            totalNumDriftPoints += totalDriftPoints
            totalInline += inline
            totalNumOfSentimentCircles += numOfSentimentCircles
    else:
        totalBeforeDict, totalExact, totalAfterDict, totalNumDriftPoints, totalInline, totalNumOfSentimentCircles = plotter(
            stockCode, saveLocation, start, end, threshold, generateDriftCircles, CDD, SCD, sentimentChange,
            thresholdSentiment, windowSize, showGraph, saveSentimentLocation, sentimentWindowSize, sentimentThreshold,
            volumeThreshold)

    totalBefore = sum(totalBeforeDict.values())
    totalAfter = sum(totalAfterDict.values())
    total = totalBefore + totalExact + totalAfter

    if saveTextLocation != "":
        sys.stdout = open(saveTextLocation, "w+")

    if printValues:
        print("Number of drifts = " + str(totalNumDriftPoints))
        print("Number of drifts not inline with low/high sentiment score = " + str(totalNumDriftPoints - total))
        print("Number of drifts inline with very low/high sentiment score = " + str(total) + '(' + str(
            round((total / totalNumDriftPoints) * 100)) + "%)")
        print("")
        print("-Number of inline drifts occurring on the same day = " + str(totalExact) + " (" + str(
            round((totalExact / total) * 100, 2)) + "%)")
        print("")
        print("-Number of inline drifts occurring early = " + str(totalBefore) + " (" + str(
            round((totalBefore / total) * 100, 2)) + "%)")

        if totalBeforeDict[-1] != 0:
            print("--Number of inline drifts occurring 1 day early = " + str(totalBeforeDict[-1]) + " (" + str(
                round((totalBeforeDict[-1] / total) * 100, 2)) + "%)")
        for i in range(2, 6):
            if totalBeforeDict[-i] != 0:
                print("--Number of inline drifts occurring " + str(i) + " days early = " + str(
                    totalBeforeDict[-i]) + " (" + str(
                    round((totalBeforeDict[-i] / total) * 100, 2)) + "%)")

        print("")

        print("-Number of inline drifts occurring late = " + str(totalAfter) + " (" + str(
            round((totalAfter / total) * 100, 2)) + "%)")

        if totalAfterDict[1] != 0:
            print("--Number of inline drifts occurring 1 day late = " + str(totalAfterDict[1]) + " (" + str(
                round((totalAfterDict[1] / total) * 100, 2)) + "%)")
        for i in range(2, 6):
            if totalAfterDict[i] != 0:
                print(
                    "--Number of inline drifts occurring " + str(i) + " days late = " + str(
                        totalAfterDict[i]) + " (" + str(
                        round((totalAfterDict[i] / total) * 100, 2)) + "%)")

        print("")
        print("Number of low/high sentiment scores inline with drifts = " + str(totalInline) + '(' + str(
            round(totalInline / totalNumOfSentimentCircles, 2) * 100) + "%)")
        print(
            "Number of low/high sentiment scores not inline with drifts = " + str(
                totalNumOfSentimentCircles - totalInline))

    if thresholdSentiment or sentimentChange:
        return [totalNumDriftPoints, str(total) + '(' + str(
            round((total / totalNumDriftPoints) * 100)) + "%)", str(totalInline) + '(' + str(
            round((totalInline / totalNumOfSentimentCircles) * 100)) + "%)", totalNumOfSentimentCircles]


if __name__ == "__main__":
    # print(plot(20, sentimentChange=True, thresholdSentiment=False, printValues=True))
    print(calculateCorrelationFunction(5, CDD="hddma", predictCD=True, considerMovement=True))
