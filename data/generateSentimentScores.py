"""
Data used from New York Times:
-abstract
-snippet
-lead_paragraph
-headline
--main
"""

import csv
import json
import nltk
import os
import shutil
from src.finbert import predict
from transformers import AutoModelForSequenceClassification


if __name__ == "__main__":

    # Parameter Declaration
    datasetName = "kdd17"  # Can be either "kdd17" or "stocknet"

    # Variable Declaration
    stockCodes = os.listdir(datasetName + "/NYT-Business")
    FinBert = AutoModelForSequenceClassification.from_pretrained("../models/FinBert", cache_dir=None, num_labels=3)


    nltk.download("punkt")

    if os.path.exists(datasetName + "/SentimentScores/NYT-Business"):
        shutil.rmtree(datasetName + "/SentimentScores/NYT-Business")

    os.mkdir(datasetName + "/SentimentScores/NYT-Business")

    for stock in stockCodes:
        os.mkdir(datasetName + "/SentimentScores/NYT-Business/" + stock)
        for date in os.listdir(datasetName + "/NYT-Business/" + stock):
            dateResults = [["Sentence", "Prediction", "sentimentScore"]]
            with open(datasetName + "/NYT-Business/" + stock + '/' + date, encoding="utf-8") as file:
                data = json.load(file)
                completed = []
                for text in [data["abstract"], data["snippet"], data["lead_paragraph"], data["headline"]["main"]]:
                    if text is not None and text != "" and not str.isspace(text) and text not in completed:
                        completed.append(text)
                        temp = predict(text, FinBert)
                        for column in temp.iterrows():
                            dateResults.append([column[1]["sentence"], column[1]["prediction"], column[1]["sentiment_score"]])
            with open(datasetName + "/SentimentScores/NYT-Business/" + stock + '/' + date[:-5] + ".csv", "w+", encoding="utf-8", newline='') as file:
                csvWriter = csv.writer(file)
                csvWriter.writerows(dateResults)
            exit()