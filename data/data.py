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

import json
import os
import shutil
import yfinance
from tqdm import tqdm

# Global variable declaration
unwantedCharacters = "!#$%\'()*+,-./:;<=>?@[\\]^_`{|}~012456789"
stopwords = []


def saveArticle(stockCode, article):
    with open("kdd17/NYT-Business/" + stockCode.upper() + '/' + article["pub_date"][0:10] + ".json", "w+", encoding="utf-8") as saveFile:
        json.dump(article, saveFile)


def compare(stockName, word):
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
        return [word for word in words.translate(str.maketrans('', '', unwantedCharacters)).lower().split() if word not in stopwords]
    return []


if __name__ == "__main__":
    # Starting with KDD17 NYT

    # Variable Declaration
    stockCodes = os.listdir("kdd17/Numerical/ourpped")
    stockNames = []

    # Parameter Declaration
    remakeFiles = True
    remakeStockNames = False

    # Get all stock codes
    for i in range(len(stockCodes)):
        stockCodes[i] = stockCodes[i][:-4]

    # Make directories and delete previously extracted articles
    if remakeFiles:
        if os.path.exists("kdd17/NYT-Business"):
            shutil.rmtree("kdd17/NYT-Business")
            os.mkdir("kdd17/NYT-Business")
        for code in stockCodes:
            os.mkdir("kdd17/NYT-Business/" + code)

    # Create stockNames.txt
    if remakeStockNames or not os.path.exists("kdd17/stockNames.txt"):
        if os.path.exists("kdd17/stockNames.txt"):
            os.remove("kdd17/stockNames.txt")
        with open("kdd17/stockNames.txt", "w+") as file:
            for code in tqdm(stockCodes):
                try:
                    file.write(yfinance.Ticker(code).info["longName"] + "\n")
                except (KeyError, TypeError):
                    file.write(code + "\n")
        print("Manual checking of stock names is now required.")
        print("Rerun script with remakeStockNames=False when stock names are checked.")

    # Get stock names
    with open("kdd17/stockNames.txt") as file:
        stockNames = file.read().split("\n")

    # Generate dictionary with stock codes and names
    stocks = {}
    for i in range(len(stockNames)):
        if '/' in stockNames[i]:
            stocks[stockCodes[i].lower()] = [c.lower() for c in stockNames[i].split('/')]
        else:
            stocks[stockCodes[i].lower()] = stockNames[i].lower()

    # Get stopwords
    with open("stopwords.txt") as stopwordsFile:
        stopwords = stopwordsFile.read().split("\n")

        # Get articles
        for fileName in tqdm(os.listdir("NYT-Business")):
            with open("NYT-Business/" + fileName, encoding="utf-8") as file:
                data = json.load(file)
                for article in data:
                    words = validate(article["abstract"]) + validate(article["snippet"]) + \
                            validate(article["lead_paragraph"]) + validate(article["headline"]["main"]) + \
                            validate(article["headline"]["kicker"]) + validate(article["headline"]["content_kicker"]) + \
                            validate(article["headline"]["print_headline"]) + validate(article["headline"]["name"]) + \
                            validate(article["headline"]["seo"]) + validate(article["headline"]["sub"])

                    if article["keywords"] is not None:
                        for keyword in article["keywords"]:
                            words.extend(validate(keyword["value"]))

                    for i, word in enumerate(words):
                        for stock in stocks.items():
                            if stock[0] == word:
                                saveArticle(stock[0], article)
                                break

                            generated = False

                            if isinstance(stock[1], list):
                                for current in stock[1]:
                                    if compare(current, word):
                                        saveArticle(stock[0], article)
                                        generated = True
                                        break
                            else:
                                if compare(stock[1], word):
                                    saveArticle(stock[0], article)
                                    break
                            if generated:
                                break

