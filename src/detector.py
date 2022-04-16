import pandas
import skmultiflow.drift_detection


def conceptDriftDetector(stream, CDD="hddma"):

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
        CDD = skmultiflow.drift_detection.hddm_a.HDDM_A()
    elif CDD == "hddmw":
        CDD = skmultiflow.drift_detection.hddm_w.HDDM_W()
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

    print(conceptDriftDetector(pandas.read_csv("../data/" + datasetName + "/Numerical/price_long_50/AAPL.csv", index_col="Date", parse_dates=["Date"])["Close"].iloc[::-1].tolist()))
