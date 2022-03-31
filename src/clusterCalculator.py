"""
Original algorithms can be found in:
https://www-m9.ma.tum.de/material/felix-klein/clustering/Methoden/Hierarchisches_Clustern_Beispiel.php
"""


import csv
import numpy
from decimal import Decimal, ROUND_HALF_UP
from itertools import groupby


if __name__ == "__main__":

    # Parameter Declaration
    distanceMetric = "tv"  # Can be either "pad" or "kl" or "hellinger" or "tv"

    # Variable Declaration
    distances = []
    file = ""

    if distanceMetric == "kl":
        file = "../results/distances/KL Distances.csv"
    elif distanceMetric == "pad":
        file = "../results/distances/Proxy A-Distances.csv"
    elif distanceMetric == "hellinger":
        file = "../results/distances/Hellinger Distances.csv"
    elif distanceMetric == "tv":
        file = "../results/distances/Total Variation Distances.csv"
    else:
        print("Parameter error: distanceMetric=" + str(distanceMetric) + " is not an accepted input")
        exit()

    with open(file) as file:
        csvReader = csv.reader(file)
        next(csvReader)  # Skip headers
        distances = list(csvReader)

        for c in range(len(distances)):
            distances[c] = list(map(float, distances[c]))

    distancesNumpy = numpy.array(distances)
    smallestDistance = numpy.min(distancesNumpy[numpy.nonzero(distancesNumpy)])
    c = 0

    while c <= 3:
        distances[c] = distances[c][4:]
        c += 1

    del distances[-5:]

    print("The finance cluster differs from the non-finance cluster by:")
    if smallestDistance == 0:
        print("Using Single Link --> " + str(Decimal(Decimal(min([c for dataset in distances for c in dataset])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))) + '%')
        print("Using Complete Link --> " + str(Decimal(Decimal(max([c for dataset in distances for c in dataset])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))) + '%')
        print("Using Average Link --> " + str(Decimal(Decimal((sum([sum(i) for i in zip(*distances)]) / 20)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))) + '%')