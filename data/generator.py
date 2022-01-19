"""
KDD17
- Numerical (X)
- New York Times
- Twitter

--------------------

Stocknet
- Numerical
- New York Times
- Twitter
"""

"""
Data extracted from New York Times:
-abstract
-snippet
-lead_paragraph
-headline
--main
--kicker
--content_kicker
--print_headline
--name
--seo
--sub
-keywords
--value

Ensured that the above contains all the important data by checking different data values from various timestamps
"""

import csv
import json
import os
import shutil
import yfinance
from tqdm import tqdm

# Global variable declaration
unwantedCharacters = "!#$%\'()*+,-./:;<=>?@[\\]^_`{|}~012456789"
stopwords = []
stockArticleCounter = {}
words = []
datasetName = ""


def saveArticle(stockCode, article):
    stockArticleCounter[stockCode.upper()] += 1

    with open(datasetName + "/NYT-Business/" + stockCode.upper() + '/' + article["pub_date"][0:10] + ".json", "w+",
              encoding="utf-8") as saveFile:
        json.dump(article, saveFile)


def compare(stockName, word):
    # If stockName has multiple words
    if " " in stockName:
        save = True
        for j, current in enumerate(stockName.split()):
            if i + j >= len(words):
                break
            if current != words[i + j]:
                save = False
                break
        if save:
            return True
        return False

    if word == stockName:
        return True

    return False


def validate(words):
    if words is not None:
        return [word for word in words.translate(str.maketrans('', '', unwantedCharacters)).lower().split() if
                word not in stopwords]
    return []


if __name__ == "__main__":
    # Parameter Declaration
    remakeFiles = True
    remakeStockNames = False
    datasetName = "kdd17"  # Can be either "kdd17" or "stocknet"
    sentimentType = "Twitter"  # Can be either "Twitter" or "NYT-Business"

    # Variable Declaration
    stockCodes = os.listdir(datasetName + "/Numerical/ourpped")
    stockNames = []


    # Get all stock codes
    for i in range(len(stockCodes)):
        stockCodes[i] = stockCodes[i][:-4]

    # Make directories and delete previously extracted articles
    if remakeFiles:
        if os.path.exists(datasetName + '/' + sentimentType):
            shutil.rmtree(datasetName + '/' + sentimentType)
            os.mkdir(datasetName + '/' + sentimentType)
        for code in stockCodes:
            os.mkdir(datasetName + '/' + sentimentType + '/' + code)

    if sentimentType == "Twitter":
        print("Handle Twitter here")

    elif sentimentType == "NYT-Business":
        # Create stockNames.txt
        if remakeStockNames or not os.path.exists(datasetName + "/stockNames.txt"):
            if os.path.exists(datasetName + "/stockNames.txt"):
                os.remove(datasetName + "/stockNames.txt")
            with open(datasetName + "/stockNames.txt", "w+") as file:
                for code in tqdm(stockCodes):
                    try:
                        file.write(yfinance.Ticker(code).info["longName"] + "\n")
                    except (KeyError, TypeError):
                        file.write(code + "\n")
            print("Manual checking of stock names is now required.")
            print("Rerun script with remakeStockNames=False when stock names are checked.")
            exit()

        # Get stock names
        with open(datasetName + "/stockNames.txt") as file:
            stockNames = file.read().split("\n")

        # Generate dictionary with stock codes and names
        stocks = {}
        for i in range(len(stockNames)):

            stockArticleCounter[stockCodes[i]] = 0

            if '/' in stockNames[i]:
                stocks[stockCodes[i].lower()] = [c.lower() for c in stockNames[i].split('/')]
            else:
                stocks[stockCodes[i].lower()] = stockNames[i].lower()

        # Get stopwords
        with open("stopwords.txt") as stopwordsFile:
            stopwords = stopwordsFile.read().split("\n")

            # Get articles
            for fileName in tqdm(os.listdir("NYT-Business")):

                if datasetName == "stocknet" and int(fileName[:-5][-4:]) < 2014:
                    continue

                with open("NYT-Business/" + fileName, encoding="utf-8") as file:
                    data = json.load(file)
                    for article in data:
                        # Get all needed data from article
                        words = validate(article["abstract"]) + validate(article["snippet"]) + \
                                validate(article["lead_paragraph"]) + validate(article["headline"]["main"]) + \
                                validate(article["headline"]["kicker"]) + \
                                validate(article["headline"]["content_kicker"]) + \
                                validate(article["headline"]["print_headline"]) + validate(article["headline"]["name"]) + \
                                validate(article["headline"]["seo"]) + validate(article["headline"]["sub"])

                        if article["keywords"] is not None:
                            for keyword in article["keywords"]:
                                words.extend(validate(keyword["value"]))

                        for i, word in enumerate(words):
                            for stock in stocks.items():
                                # Check if stock code in article
                                if stock[0] == word:
                                    saveArticle(stock[0], article)
                                    continue

                                generated = False

                                # Check if stock name in article

                                if isinstance(stock[1], list):
                                    for current in stock[1]:
                                        if compare(current, word):
                                            saveArticle(stock[0], article)
                                            generated = True
                                            break
                                else:
                                    if compare(stock[1], word):
                                        saveArticle(stock[0], article)
                                        continue
                                if generated:
                                    continue
        with open(datasetName + "/NYT-Business/stockArticleCounter.csv", "w+", newline='') as stockArticleCounterFile:
            writer = csv.writer(stockArticleCounterFile)
            writer.writerow(["Stock", "Number of Articles"])
            for stock, count in stockArticleCounter.items():
                writer.writerow([stock, count])
    else:
        print(sentimentType + " is not a valid sentimentType parameter value")
