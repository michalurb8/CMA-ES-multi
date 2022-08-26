import numpy as np

gaussian2 = lambda p, xmu=0, ymu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2)/4)
gaussian3 = lambda p, xmu=0, ymu=0, zmu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2+(p[2]-zmu)**2)/4)

criteriumList = [
    [
        lambda p : 0.01*(p[0]/7)**2 + ((p[1]+1)/7-((p[0]+1)/7)**2)**2,
        lambda p : gaussian2(p, 2, 4),
        "rosenbrock i punkt",
    ],
    [
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        "dwa zrównoważone podwójne gaussy",
    ],
    [
        lambda p : gaussian2(p, 0, 0),
        lambda p : (p[0]+p[1])/10,
        "punkt plus liniowa",
    ],
    [
        lambda p : gaussian2(p, 0, 2) + 0.8*gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 2, 0) + 0.8*gaussian2(p, -2, 0),
        "dwa nierówne gaussy równo skrzyżowane",
    ],
    [
        lambda p : gaussian2(p, 2, 2),
        lambda p : gaussian2(p, -2, 2),
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        "kwadrat",
    ],
    [
        lambda p : gaussian2(p, 0, 3) + gaussian2(p, 0, -1),
        lambda p : gaussian2(p, 3, 0) + gaussian2(p, -1, 0),
        "dwa równe podwójne gaussy nierówno ułożone",
    ],
    [
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 0, 3),
        "trzy punkty równoramiennie",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        lambda p : (p[0]+p[1])/10,
        "podwójny równy gauss plus punkt plus liniowa",
    ],
    [
        lambda p : gaussian2(p, 2, 0),
        lambda p : gaussian2(p, -2, 0),
        lambda p : gaussian2(p, 0, 0),
        "trzy współniniowe punkty",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2),
        "równa gwiazda",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2),
        "gwiazda z jednym czubkiem bardziej",
    ],
    [
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2.2) + gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, 0),
        "nierówna gwiazda + punkt",
    ],
    [
        lambda p : gaussian2(p, 4, 2),
        lambda p : gaussian2(p, -3, -1),
        "dwa punkty",
    ],
    [
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        lambda p : gaussian2(p),
        "dwa zrównoważone podwójne gaussy plus punkt",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0.1),
        "podwójny równy gauss plus punkt krzywo",
    ],
    [
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        "podwójny równy gauss plus punkt",
    ],
]

criteria = criteriumList[0][:-1]
funName = criteriumList[0][-1]