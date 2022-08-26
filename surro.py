from posixpath import dirname
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib
import pickle
from sys import stdout

from functions import criteria, funName

COLORS = ['red', 'black', 'green']
COLOR_NUM = len(COLORS)
INF = float('inf')

def getBackgroundData(rang = 30):
    res = 2*rang + 1

    fileName = str(funName) + "_" + str(rang)
    print("fileName", fileName)
    fileExists = False

    dirName = "cached"
    if not os.path.isdir(".\\" + dirName):
        os.mkdir(dirName)

    wholePath = ".\\" + dirName + "\\" + fileName
    fileExists = os.path.isfile(wholePath)

    if fileExists:
        print("File found!")
        with open(wholePath, 'rb') as file:
            xGrid, yGrid, rGrid, levels = pickle.load(file)
        return (xGrid, yGrid, rGrid, levels)
    else:
        print("File not found. Generating points...")

        points = generatePoints(res)
        print("Points generated. Calculating ranks...")

        calcRank(points)
        print("Ranks calculated. Clearing...")
        points = cleared(points)
        print("Points cleared. Drawing...")

        maxRank = max([point[2] for point in points])
        print("maxrank:", maxRank)
        xGrid, yGrid, rGrid = np.array(list(zip(*points)))
        step = int(np.power(maxRank, 0.33))
        levels = [*np.array(range(step)), *(np.array(range(step))*(step-1) + step), *(np.array(range(step))*step*(step-1) + step*step)]

        print("Saving to file...")
        file = open(wholePath, 'wb')
        pickle.dump([xGrid, yGrid, rGrid, levels], file)

        with open(wholePath, 'rb') as file:
            xGrid, yGrid, rGrid, levels = pickle.load(file)
        return xGrid, yGrid, rGrid, levels

def cleared(points): # change the structure of points list. Removing nesting and discarding criterias' values
    points = [[point[0][0], point[0][1], point[2]] for point in points]
    return points

def generatePoints(res):
    rang = res // 2
    points = []
    for x in range(res):
        for y in range(res):
            # newX = np.random.uniform()*10 - 5; newY = np.random.uniform()*10 - 5; # uniform -5 to 5
            newX = np.random.normal()*2; newY = np.random.normal()*2; # normal -5 to 5
            crit = np.array([criterium((newX, newY)) for criterium in criteria])
            newPoint = [np.array([newX, newY]), crit, INF]
            points.append(newPoint)
    return points

def calcRank(points):
    currentRank = 0
    while INF in [point[2] for point in points]:
        stdout.write(f"\rCurrently calculated rank: {currentRank}")
        stdout.flush()
        for currentPoint in points:
            if currentPoint[2] < currentRank: continue
            for otherPoint in points:
                if otherPoint[2] >= currentRank and isDominated(currentPoint, otherPoint):
                    break
            else:
                currentPoint[2] = currentRank
        currentRank += 1
    print()

def _drawBG(ax, res = 101):
    x_ax = np.linspace(-1, 1, res)
    y_ax = np.linspace(-1, 1, res)

    xGrid, yGrid = np.meshgrid(x_ax, y_ax)

    for index, criterium in enumerate(criteria):
        z = criterium((xGrid, yGrid))
        z = z - np.amin(z)
        z = z / np.amax(z)
        z = 10 * z
        ax.contour(xGrid, yGrid, z, levels=list(np.array(list(range(40)))/4), colors=COLORS[index%COLOR_NUM], alpha=0.2)

def isDominated(p1, p2):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True

def hashName(name):
    result = hashlib.md5(name.encode()).hexdigest()
    return result

if __name__ == "__main__":
    getBackgroundData()