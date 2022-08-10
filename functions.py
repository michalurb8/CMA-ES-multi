import numpy as np

gaussian2 = lambda p, xmu=0, ymu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2)/4)
gaussian3 = lambda p, xmu=0, ymu=0, zmu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2+(p[2]-zmu)**2)/4)

criteriumList = [

    [ # gwiazda
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2.2) + gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, 0)
    ],

    [ # gwiazda
        lambda p : gaussian2(p, -2, 2) + 1.01*gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, -2, -2),
    ],

    [ # dwa hantle równowaga plus punkt
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        lambda p : gaussian2(p)
    ],
    [ # równe hantle plus nierówny punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0.1),
    ],
    [ # skrzyżowane nierówne hantle
        lambda p : gaussian2(p, 0, 2) + 0.8*gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 2, 0) + 0.8*gaussian2(p, -2, 0),
    ],
    [ # trójkąt równoramienny
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 0, 3),
    ],
    [ # dwa hantle równowaga
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
    ],

    [ # dwa punkty
        lambda p : gaussian2(p, 4, 2),
        lambda p : gaussian2(p, -3, -1),
    ],
 [ # rosenbrock i punkt
        lambda p : 0.01*(p[0]-1)**2 + (p[1]-p[0]**2)**2,
        lambda p : gaussian2(p, 1, 2)
    ],
 [ # równe hantle plus równy punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
    ],

    [ # trzy współniniowe
        lambda p : gaussian2(p, 2, 0),
        lambda p : gaussian2(p, -2, 0),
        lambda p : gaussian2(p, 0, 0),
    ],


    [ # nierówno skrzyżowane hantle
        lambda p : gaussian2(p, 0, 3) + gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 3, 0) + gaussian2(p, -2, 0),
    ],

    [ # linia plus punkt
        lambda p : gaussian2(p, 0, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [ # linia plus równe hantle plus punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [ # kwadrat
        lambda p : gaussian2(p, 2, 2),
        lambda p : gaussian2(p, -2, 2),
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
    ],


]

criteria = criteriumList[0]