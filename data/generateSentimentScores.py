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
import logging
import nltk
import os
import shutil
import sys
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(1, "../src")
from finbert import predict


def useFinBert(article):
    for text in [article["abstract"], article["headline"]["main"]]:
        if text is not None and text != "" and not str.isspace(text):
            for column in predict(text, FinBert, tokenizer=tokenizer).iterrows():
                dataResults.append([row[0], column[1]["sentence"], column[1]["prediction"], column[1]["sentiment_score"]])


if __name__ == "__main__":

    # Parameter Declaration
    datasetName = "stocknet"  # Can be either "kdd17" or "stocknet"
    fromStart = True

    # Variable Declaration
    stockCodes = os.listdir(datasetName + "/NYT-Business/ourpped")
    FinBert = AutoModelForSequenceClassification.from_pretrained("../models/FinBert", cache_dir=None, num_labels=3).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # nltk.download("punkt")
    logging.disable()
    if fromStart:
        if os.path.exists(datasetName + "/SentimentScores/NYT-Business"):
            shutil.rmtree(datasetName + "/SentimentScores/NYT-Business")

        os.mkdir(datasetName + "/SentimentScores/NYT-Business")

    completed = os.listdir(datasetName + "/SentimentScores/NYT-Business")

    for stock in os.listdir(datasetName + "/NYT-Business/ourpped"):
        print("Doing " + stock)
        if stock in completed:
            continue
        start = time.process_time()
        dataResults = [["Date", "Sentence", "Prediction", "sentimentScore"]]
        with open(datasetName + "/NYT-Business/ourpped/" + stock, encoding="utf-8") as file:
            csvReader = csv.reader(file)
            next(csvReader)  # Skip headers
            csvReader = list(csvReader)
            for row in tqdm(csvReader):

                data = json.loads(row[1])
                if type(data) is list:
                    for article in data:
                        useFinBert(article)
                else:
                    useFinBert(data)

        with open(datasetName + "/SentimentScores/NYT-Business/" + stock, "w+", encoding="utf-8", newline='') as file:
            csvWriter = csv.writer(file)
            csvWriter.writerows(dataResults)
