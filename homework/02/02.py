from random import randint
import numpy as np


citys = [
    (0, 3), (0, 0),
    (0, 2), (0, 1),
    (1, 0), (1, 3),
    (2, 0), (2, 3),
    (3, 0), (3, 3),
    (3, 1), (3, 2)
]


def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def pathLength(path):
    dist = 0
    plen = len(path)
    for i in range(plen):
        c1 = citys[path[i]]
        c2 = citys[path[(i + 1) % plen]]  
        dist += distance(c1, c2)
    return dist


def neighbor(path):
    new_path = path.copy()
    i = randint(0, len(path) - 1)
    j = randint(0, len(path) - 1)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def hillClimbing(path, pathLength, neighbor, max_fail):
    print("start: ", pathLength(path))
    fail = 0
    gens = 0
    while True:
        new_path = neighbor(path)
        if pathLength(new_path) < pathLength(path):
            path = new_path
            gens += 1
            print(gens, ':', pathLength(path), path)
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                print("solution: ", pathLength(path))
                return path

initial_path = list(range(len(citys)))
np.random.shuffle(initial_path)


hillClimbing(initial_path, pathLength, neighbor, max_fail=10000)
