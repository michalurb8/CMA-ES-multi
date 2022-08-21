import numpy as np
import matplotlib.pyplot as plt

from functions import criteria, funName

COLORS = ['red', 'black', 'green']
COLOR_NUM = len(COLORS)
INF = float('inf')

def func():
    fileName = hashName(funName)
    fileExists = False

    # check if file exists TODO

    if fileExists:
        print("File found!")
        #LOAD from file, then return TODO
    print("File not found, calculating and saving to file...")

    rang = 50
    res = 2*rang + 1


    scale = 5

    fig, ax = plt.subplots()
    _drawBG(ax, scale)

    points = generatePoints(res, scale)
    print("Points generated")

    calcRank(points)
    points = cleared(points)
    print("Ranks calculated")

    maxRank = max([point[2] for point in points])
    print("maxrank:", maxRank)
    xGrid, yGrid, rGrid = np.array(list(zip(*points)))
    step = int(np.power(maxRank, 0.33))
    print("maxstep:", step*step*(step-1) + step*step)
    levels = [*np.array(range(step)), *(np.array(range(step))*(step-1) + step), *(np.array(range(step))*step*(step-1) + step*step)]
    # save to file TODO
    ax.tricontour(xGrid, yGrid, rGrid, linewidths = 2, levels=levels)
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))

    print("Points drawing...")
    plt.show()

def cleared(points): # change the structure of points list. Removing nesting and discarding criterias' values
    points = [[point[0][0], point[0][1], point[2]] for point in points]
    return points

def generatePoints(res, scale):
    points = []
    for _ in range(res):
        for _ in range(res):
            newX = np.random.uniform()*2 - 1
            newY = np.random.uniform()*2 - 1
            crit = np.array([criterium((newX*scale, newY*scale)) for criterium in criteria])
            newPoint = [np.array([newX, newY]), crit, INF]
            points.append(newPoint)
    return points

def calcRank(points):
    currentRank = 0
    i = 0
    while INF in [point[2] for point in points]:
        i += 1
        for currentPoint in points:
            if currentPoint[2] < currentRank: continue
            for otherPoint in points:
                if otherPoint[2] >= currentRank and isDominated(currentPoint, otherPoint):
                    break
            else:
                currentPoint[2]=currentRank
        currentRank += 1

def _drawBG(ax, scale, res = 101):
    x_ax = np.linspace(-1, 1, res)
    y_ax = np.linspace(-1, 1, res)

    xGrid, yGrid = np.meshgrid(x_ax, y_ax)

    for index, criterium in enumerate(criteria):
        z = criterium((xGrid*scale, yGrid*scale))
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
    return int(abs(hash(name)))

if __name__ == "__main__":
    func()