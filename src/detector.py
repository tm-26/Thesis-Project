import pandas
import skmultiflow.drift_detection


def detector(stream):
    hddma = skmultiflow.drift_detection.hddm_a.HDDM_A()
    driftPoints = []
    for i in range(len(stream)):
        hddma.add_element(stream[i])
        if hddma.detected_change():
            driftPoints.append(i)
    return driftPoints


if __name__ == "__main__":
    # Parameter Declaration
    datasetName = "kdd17"  # Can be either "kdd17" or "stocknet"

    print(detector(pandas.read_csv("../data/" + datasetName + "/Numerical/price_long_50/AAPL.csv", index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1].tolist()))
