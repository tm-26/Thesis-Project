"""
Data used from New York Times:
-abstract
-snippet
-lead_paragraph
-headline
--main
"""

import json
import nltk
import os
from src.finbert import predict
from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("../models/FinBert", num_labels=3, cache_dir=None)
#
# output_dir = "tempOutput"
#
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
#
#
# output = "predictions.csv"
# predict(text, model, write_to_csv=True, path=os.path.join(output_dir, output))


def validate(text):
    if text is not None or text != "" or not str.isspace(text):
        return text


if __name__ == "__main__":

    # Parameter Declaration
    datasetName = "kdd17"  # Can be either "kdd17" or "stocknet"

    nltk.download("punkt")

    for stock in os.listdir(datasetName + "/NYT-Business"):
        for date in os.listdir(datasetName + "/NYT-Business/" + stock):
            with open(datasetName + "/NYT-Business/" + stock + '/' + date, encoding="utf-8") as file:
                data = json.load(file)
                for text in [data["abstract"], data["snippet"], data["lead_paragraph"], data["headline"]["main"]]:
