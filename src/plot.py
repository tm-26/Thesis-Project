import matplotlib.pyplot
import os
import pandas
import sys
from statistics import mean
from detector import conceptDriftDetector
from tqdm import tqdm


def plot(stockCode):
    # Getting numerical first
    numerical = pandas.read_csv("../data/kdd17/Numerical/price_long_50/" + stockCode + ".csv", header=0).loc[::-1]
    numerical["Date"] = pandas.to_datetime(numerical["Date"])
    numerical = numerical.loc[(numerical["Date"] > start) & (numerical["Date"] <= end)]

    # Getting sentiment next
    sentiment = pandas.read_csv("../data/kdd17/SentimentScores/NYT-Business/" + stockCode + ".csv", header=0).loc[::-1]
    sentiment["Date"] = pandas.to_datetime(sentiment["Date"])
    sentiment = sentiment.loc[(sentiment["Date"] > start) & (sentiment["Date"] <= end)]

    sentimentScores = {}
    for index, row in sentiment.iterrows():
        if row["Date"] in sentimentScores:
            sentimentScores[row["Date"]].append(row["sentimentScore"])
        else:
            sentimentScores[row["Date"]] = [row["sentimentScore"]]

    # Plotting
    figure, axes = matplotlib.pyplot.subplots()
    axes.plot_date(numerical["Date"], numerical["Adj Close"], '-')

    axes.tick_params(axis='x', labelsize=9)

    circleFormat = ""
    driftPoints = conceptDriftDetector(numerical["Adj Close"].tolist(), CDD)

    for i in range(len(driftPoints)):
        driftPoints[i] = numerical["Date"][driftPoints[i]]

    # For statistical purposes
    exact = 0
    before = {-1: 0, -2: 0, -3: 0, -4: 0, -5: 0}
    after = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    inline = 0
    numOfSentimentCircles = 0

    for date in list(sentimentScores.keys()):
        current = mean(sentimentScores[date])

        if -threshold >= current:
            circleFormat = "or"
            numOfSentimentCircles += 1
        elif threshold <= current:
            circleFormat = "og"
            numOfSentimentCircles += 1
        else:
            continue

        # Needs to be done to account for days when stock market is not open
        while True:
            try:
                axes.plot_date(date, numerical.loc[numerical["Date"] == date, "Adj Close"].iloc[0], circleFormat,
                               fillstyle="none", ms=5.0)
            except IndexError:
                date = date + pandas.DateOffset(days=1)
            else:
                break

        driftPoints.sort(key=lambda i: abs(i - date))
        incremented = False
        for driftPoint in driftPoints:
            dateDifference = (driftPoint - date).days

            if abs(dateDifference) > 5:
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

    if saveLocation != "":
        matplotlib.pyplot.savefig(saveLocation)

    matplotlib.pyplot.show()

    return before, exact, after, len(driftPoints), inline, numOfSentimentCircles


if __name__ == "__main__":
    # Parameter Declaration
    stockCode = "all"
    saveLocation = ""
    start = "2007-1-1"  # YYYY-MM-DD
    end = "2017-12-1"  # YYYY-MM-DD
    threshold = 0.8
    generateDriftCircles = True
    saveTextLocation = "../results/sentiment in relation to concept drift/EDDM.txt"
    CDD = "eddm"

    # Variable Declaration
    totalBeforeDict = {-1: 0, -2: 0, -3: 0, -4: 0, -5: 0}
    totalExact = 0
    totalAfterDict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    totalNumDriftPoints = 0
    totalInline = 0
    totalNumOfSentimentCircles = 0

    if stockCode == "all":
        for stockCode in tqdm(os.listdir("../data/kdd17/Numerical/ourpped")):
            before, exact, after, totalDriftPoints, inline, numOfSentimentCircles = plot(stockCode[:-4])

            for i in range(1, 6):
                totalBeforeDict[-i] += before[-i]
                totalAfterDict[i] += after[i]

            totalExact += exact
            totalNumDriftPoints += totalDriftPoints
            totalInline += inline
            totalNumOfSentimentCircles += numOfSentimentCircles
    else:
        totalBeforeDict, totalExact, totalAfterDict, totalNumDriftPoints, totalInline, totalNumOfSentimentCircles = plot(stockCode)

    totalBefore = sum(totalBeforeDict.values())
    totalAfter = sum(totalAfterDict.values())
    total = totalBefore + totalExact + totalAfter

    if saveTextLocation != "":
        sys.stdout = open(saveTextLocation, "w+")

    print("Number of drifts not inline with low/high sentiment score = " + str(totalNumDriftPoints - total))
    print("Number of drifts inline with very low/high sentiment score = " + str(total))
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
            print("--Number of inline drifts occurring " + str(i) + " days early = " + str(totalBeforeDict[-i]) + " (" + str(
                round((totalBeforeDict[-i] / total) * 100, 2)) + "%)")

    print("")

    print("-Number of inline drifts occurring late = " + str(totalAfter) + " (" + str(
        round((totalAfter / total) * 100, 2)) + "%)")

    if totalAfterDict[1] != 0:
        print("--Number of inline drifts occurring 1 day late = " + str(totalAfterDict[1]) + " (" + str(
            round((totalAfterDict[1] / total) * 100, 2)) + "%)")
    for i in range(2, 6):
        if totalAfterDict[i] != 0:
            print("--Number of inline drifts occurring " + str(i) + " days late = " + str(totalAfterDict[i]) + " (" + str(
                round((totalAfterDict[i] / total) * 100, 2)) + "%)")

    print("")
    print("Number of low/high sentiment scores inline with drifts = " + str(totalInline))
    print("Number of low/high sentiment scores not inline with drifts = " + str(totalNumOfSentimentCircles - totalInline))
