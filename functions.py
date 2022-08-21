import numpy as np

gaussian2 = lambda p, xmu=0, ymu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2)/4)
gaussian3 = lambda p, xmu=0, ymu=0, zmu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2+(p[2]-zmu)**2)/4)

criteriumList = [
    [
        lambda p : gaussian2(p, 2, 2),
        lambda p : gaussian2(p, -2, 2),
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        "kwadrat",
    ],
    [
        lambda p : gaussian2(p, 0, 0),
        lambda p : (p[0]+p[1])/10,
        "linia plus punkt",
    ],
    [
        lambda p : gaussian2(p, 0, 3) + gaussian2(p, 0, -1),
        lambda p : gaussian2(p, 3, 0) + gaussian2(p, -1, 0),
        "nierówno skrzyżowane hantle",
    ],
    [
        lambda p : 0.01*(p[0]-1)**2 + (p[1]-p[0]**2)**2,
        lambda p : gaussian2(p, 1, 2),
        "rosenbrock i punkt",
    ],
    [
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 0, 3),
        "trójkąt równoramienny",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        lambda p : (p[0]+p[1])/10,
        "linia plus równe hantle plus punkt",
    ],
    [
        lambda p : gaussian2(p, 2, 0),
        lambda p : gaussian2(p, -2, 0),
        lambda p : gaussian2(p, 0, 0),
        "trzy współniniowe",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2),
        "równa gwiazda",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2),
        "gwiazda",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2.2) + gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, 0),
        "gwiazda + punkt",
    ],
    [
        lambda p : gaussian2(p, 4, 2),
        lambda p : gaussian2(p, -3, -1),
        "dwa punkty",
    ],
    [
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        "dwa hantle równowaga",
    ],
    [
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        lambda p : gaussian2(p),
        "dwa hantle równowaga plus punkt",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0.1),
        "równe hantle plus nierówny punkt",
    ],
    [
        lambda p : gaussian2(p, 0, 2) + 0.8*gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 2, 0) + 0.8*gaussian2(p, -2, 0),
        "skrzyżowane nierówne hantle",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        "równe hantle plus równy punkt",
    ],
]

criteria = criteriumList[0][:-1]
funName = criteriumList[0][-1]