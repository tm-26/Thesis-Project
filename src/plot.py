import matplotlib.pyplot
import pandas
from statistics import mean
from detector import conceptDriftDetector

if __name__ == "__main__":
    # Parameter Declaration
    stockCode = "AAPL"
    saveLocation = "../results/graphs/" + "stockCode"
    start = "2007-1-1"  # YYYY-MM-DD
    end = "2017-12-1"  # YYYY-MM-DD
    threshold = 0.8
    generateDriftCircles = True

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
    bottom, top = matplotlib.pyplot.ylim()

    axes.tick_params(axis='x', labelsize=9)

    circleFormat = ""

    for date in list(sentimentScores.keys()):
        current = mean(sentimentScores[date])

        if -threshold >= current:
            circleFormat = "or"
        elif threshold <= current:
            circleFormat = "og"
        else:
            continue

        # Needs to be done to account for days when stock market is not open
        while True:
            try:
                axes.plot_date(date, numerical.loc[numerical["Date"] == date, "Adj Close"].iloc[0], circleFormat, fillstyle="none", ms=5.0)
            except IndexError:
                date = date + pandas.DateOffset(days=1)
            else:
                break

    if generateDriftCircles:

        for driftPoint in conceptDriftDetector(numerical["Adj Close"].tolist()):
            date = numerical["Date"][driftPoint]
            axes.plot_date(date, numerical.loc[numerical["Date"] == date, "Adj Close"].iloc[0], 'o',
                           fillstyle="none", ms=5.0, color="black")

    axes.title.set_text(stockCode)
    axes.set_xlabel("Time")
    axes.set_ylabel("Adj Close Price ($)")

    if saveLocation != "":
        matplotlib.pyplot.savefig(saveLocation)

    matplotlib.pyplot.show()

    print("Graph generated")
