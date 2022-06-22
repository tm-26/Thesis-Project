import pandas
import skmultiflow.drift_detection
from datetime import timedelta
from detecta import detect_cusum
from statistics import mean
from financialCDD import minps, mySD, myTanDD


def sentimentChangeDetector(stream, SCD="cusum", changeConfidence=1):

    if SCD == "cusum":
        return detect_cusum(stream, changeConfidence, show=False)[0].tolist()
    elif SCD == "adwin":
        driftPoints =[]
        SCD = skmultiflow.drift_detection.adwin.ADWIN()

        for i in range(len(stream)):
            SCD.add_element(stream[i])
            if SCD.detected_change():
                driftPoints.append(i)
        return driftPoints


def stockChangeDetector(data, CDD="hddma", driftConfidence=0.001, SCD="cusum",
                        changeConfidence=1, typeOfReturn="all"):

    priceSeries = pandas.Series(data["Adj-Close Price"].values, index=data["Date"]).dropna()
    driftPoints = conceptDriftDetector(priceSeries.tolist(), CDD, driftConfidence)

    sentimentSeries = pandas.Series(data["Sentiment"].values, index=data["Date"]).dropna()
    changePoints = sentimentChangeDetector(sentimentSeries.tolist(), SCD, changeConfidence)

    conceptDriftDays = []

    if typeOfReturn == "all":
        for point in driftPoints:
            conceptDriftDays.append(priceSeries.index[point])

        for point in changePoints:

            # Needs to be done to account for days when stock market is not open
            currentDate = sentimentSeries.index[point]
            while True:
                if currentDate in priceSeries.index:
                    conceptDriftDays.append(currentDate)
                    break
                currentDate = currentDate + pandas.DateOffset(days=1)

        conceptDriftDays.sort()
        return list(dict.fromkeys(conceptDriftDays))

def conceptDriftDetector(stream, CDD="hddma", driftConfidence=0.001):
    driftPoints = []

    if CDD == "eddm":
        EDDM = skmultiflow.drift_detection.eddm.EDDM()

        previous = 0
        for i in range(len(stream)):
            if previous < stream[i]:
                EDDM.add_element(1)
            else:
                EDDM.add_element(0)

            previous = stream[i]

            if EDDM.detected_change():
                driftPoints.append(i)
        return driftPoints

    elif CDD == "hddma":
        CDD = skmultiflow.drift_detection.hddm_a.HDDM_A(driftConfidence)
    elif CDD == "hddmw":
        CDD = skmultiflow.drift_detection.hddm_w.HDDM_W()
    elif CDD == "minps":
        CDD = minps.MINPS(20)
    elif CDD == "mysd":
        CDD = mySD.mySDDD(20)
    elif CDD == "mytandd":
        CDD = myTanDD.myTanDD(20)
    elif CDD == "ph":
        CDD = skmultiflow.drift_detection.page_hinkley.PageHinkley()

    for i in range(len(stream)):
        CDD.add_element(stream[i])
        if CDD.detected_change():
            driftPoints.append(i)
    return driftPoints


if __name__ == "__main__":
    # Parameter Declaration
    datasetName = "kdd17"  # Can be either "kdd17" or "stocknet"
    stockCode = "AAPL"
    startDate = "2007-01-01"
    endDate = "2017-01-01"

    numerical = pandas.read_csv("../data/" + datasetName + "/Numerical/price_long_50/AAPL.csv", index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1]

    sentiment = pandas.read_csv("../data/" + datasetName +"/SentimentScores/NYT-Business/" + stockCode + ".csv", header=0)
    sentiment["Date"] = pandas.to_datetime(sentiment["Date"])

    sentimentScores = {}
    for index, row in sentiment.iterrows():
        if row["Date"] in sentimentScores:
            sentimentScores[row["Date"]].append(row["sentimentScore"])
        else:
            sentimentScores[row["Date"]] = [row["sentimentScore"]]

    startDate = pandas.Timestamp(startDate)
    endDate = pandas.Timestamp(endDate)
    allDates = []
    allPrices = []
    allSentiment = []
    delta = endDate - startDate

    for counter in range(delta.days + 1):
        date = startDate + timedelta(days=counter)

        allDates.append(date)

        if date in numerical.index:
            allPrices.append(numerical[date])
        else:
            allPrices.append(None)

        if date in sentimentScores:
            allSentiment.append(mean(sentimentScores[date]))
        else:
            allSentiment.append(None)

    dataFrame = pandas.DataFrame({"Date": allDates, "Adj-Close Price": allPrices, "Sentiment": allSentiment})
    print(stockChangeDetector(dataFrame))