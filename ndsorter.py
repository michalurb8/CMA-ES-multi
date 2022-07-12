from typing import List
import numpy as np
INF = float('inf')

def _isDominated(p1, p2):
    for v1, v2 in zip(p1[1], p2[1]):
        if v1 <= v2:
            return False
    return True

def calcRank(points: List) -> None:
    # points is a list of (points) lists of (list of coordinates, list of objectives and rank):
    # [[[c1, c2], [o1, o2, o3], rank], ...]
    currentRank = 0
    i = 0
    while INF in [point[2] for point in points]:
        i += 1
        for currentPoint in points:
            if currentPoint[2] < currentRank: continue
            for otherPoint in points:
                if otherPoint[2] >= currentRank and _isDominated(currentPoint, otherPoint):
                    break
            else:
                currentPoint[2]=currentRank
        currentRank += 1