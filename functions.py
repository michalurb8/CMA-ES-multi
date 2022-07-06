import numpy as np

gaussian2 = lambda x, y, xmu=0, ymu=0 : -np.exp(-((x-xmu)**2+(y-ymu)**2)/4)

criteriumList = [
    [
        # dwa punkty
        lambda p : gaussian2(p[0],p[1], 0,3),
        lambda p : gaussian2(p[0],p[1], 3,0),
    ],

    [
        # dwa hantle równowaga plus punkt
        lambda p : 0.6*gaussian2(p[0],p[1], 0,3) + 0.4*gaussian2(p[0],p[1], 0,-2),
        lambda p : 0.6*gaussian2(p[0],p[1], 3,0) + 0.4*gaussian2(p[0],p[1], -2,0),
        lambda p : gaussian2(p[0],p[1])
    ],

    [
        # dwa hantle równowaga
        lambda p : 0.6*gaussian2(p[0],p[1], 0,3) + 0.4*gaussian2(p[0],p[1], 0,-2),
        lambda p : 0.6*gaussian2(p[0],p[1], 3,0) + 0.4*gaussian2(p[0],p[1], -2,0),
    ],

    [
        # rosenbrock i punkt
        lambda p : 0.01*(p[0]-1)**2 + (p[1]-p[0]**2)**2,
        lambda p : gaussian2(p[0],p[1])
    ],

    [
        # sinusy
        lambda p : np.sin(0.2*p[0])*(0.2*p[0]-1)*(0.2*p[1]-2),
        lambda p : np.cos(0.2*p[1])*0.2*p[0]
    ],

    [
        # równe hantle plus punkt
        lambda p : gaussian2(p[0],p[1], 2,2) + gaussian2(p[0],p[1], 2,-2),
        lambda p : gaussian2(p[0],p[1], -2,0.1),
    ],

    [
        # nierówno skrzyżowane hantle
        lambda p : gaussian2(p[0],p[1], 0,3) + gaussian2(p[0],p[1], 0,-2),
        lambda p : gaussian2(p[0],p[1], 3,0) + gaussian2(p[0],p[1], -2,0),
    ],

    [
        # linia plus punkt
        lambda p : gaussian2(p[0],p[1], 0,0),
        lambda p : (p[0]+p[1])/10
    ],

    [
        # linia plus równe hantle plus punkt
        lambda p : gaussian2(p[0],p[1], 2,2) + gaussian2(p[0],p[1], 2,-2),
        lambda p : gaussian2(p[0],p[1], -2,0),
        lambda p : (p[0]+p[1])/10
    ],

    [
        # kwadrat
        lambda p : gaussian2(p[0],p[1],2,2),
        lambda p : gaussian2(p[0],p[1],-2,2),
        lambda p : gaussian2(p[0],p[1],-2,-2),
        lambda p : gaussian2(p[0],p[1],2,-2),
    ],

    [
        # trójkąt równoramienny
        lambda p : gaussian2(p[0],p[1],-2,-2),
        lambda p : gaussian2(p[0],p[1],2,-2),
        lambda p : gaussian2(p[0],p[1],0,3),
    ],

    [
        # trzy współniniowe
        lambda p : gaussian2(p[0],p[1],2,0),
        lambda p : gaussian2(p[0],p[1],-2,0),
        lambda p : gaussian2(p[0],p[1],0,0),
    ],

    [
        # skrzyżowane nierówne hantle
        lambda p : 0.501*gaussian2(p[0],p[1], 0,2) + 0.499*gaussian2(p[0],p[1], 0,-2),
        lambda p : 0.501*gaussian2(p[0],p[1], 2,0) + 0.499*gaussian2(p[0],p[1], -2,0),
    ],
]

criteria = criteriumList[0]