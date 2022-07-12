import numpy as np

gaussian2 = lambda p, xmu=0, ymu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2)/4)
gaussian3 = lambda p, xmu=0, ymu=0, zmu=0 : -np.exp(-((p[0]-xmu)**2+(p[1]-ymu)**2+(p[2]-zmu)**2)/4)

criteriumList = [

    [
        # rosenbrock i punkt
        lambda p : 0.01*(p[0]-1)**2 + (p[1]-p[0]**2)**2,
        lambda p : gaussian2(p, 1, 1)
    ],

    [
        # dwa hantle równowaga
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
    ],

    [
        # równe hantle plus równy punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
    ],

    [
        # równe hantle plus nierówny punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0.1),
    ],

    [
        # dwa hantle równowaga plus punkt
        lambda p : 0.6*gaussian2(p, 0, 3) + 0.4*gaussian2(p, 0, -2),
        lambda p : 0.6*gaussian2(p, 3, 0) + 0.4*gaussian2(p, -2, 0),
        lambda p : gaussian2(p)
    ],

    [
        # dwa punkty
        lambda p : gaussian2(p, 0, 3),
        lambda p : gaussian2(p, 3, 0),
    ],

    [
        # rosenbrock i punkt
        lambda p : 0.01*(p[0]-1)**2 + (p[1]-p[0]**2)**2,
        lambda p : gaussian2(p)
    ],

    [
        # sinusy
        lambda p : np.sin(0.2*p[0])*(0.2*p[0]-1)*(0.2*p[1]-2),
        lambda p : np.cos(0.2*p[1])*0.2*p[0]
    ],

    [
        # nierówno skrzyżowane hantle
        lambda p : gaussian2(p, 0, 3) + gaussian2(p, 0, -2),
        lambda p : gaussian2(p, 3, 0) + gaussian2(p, -2, 0),
    ],

    [
        # linia plus punkt
        lambda p : gaussian2(p, 0, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [
        # linia plus równe hantle plus punkt
        lambda p : gaussian2(p, 2, 2) + gaussian2(p, 2, -2),
        lambda p : gaussian2(p, -2, 0),
        lambda p : (p[0]+p[1])/10
    ],

    [
        # kwadrat
        lambda p : gaussian2(p, 2, 2),
        lambda p : gaussian2(p, -2, 2),
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
    ],

    [
        # trójkąt równoramienny
        lambda p : gaussian2(p, -2, -2),
        lambda p : gaussian2(p, 2, -2),
        lambda p : gaussian2(p, 0, 3),
    ],

    [
        # trzy współniniowe
        lambda p : gaussian2(p, 2, 0),
        lambda p : gaussian2(p, -2, 0),
        lambda p : gaussian2(p, 0, 0),
    ],

    [
        # skrzyżowane nierówne hantle
        lambda p : 0.501*gaussian2(p, 0, 2) + 0.499*gaussian2(p, 0, -2),
        lambda p : 0.501*gaussian2(p, 2, 0) + 0.499*gaussian2(p, -2, 0),
    ],
]

criteria = criteriumList[0]


if __name__ == "__main__":
    # test test
    print([crit([1,2]) for crit in criteria])