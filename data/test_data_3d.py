import numpy as np
from numpy import sin, cos, arctan2, pi, linspace, array
from shapely.geometry import LineString, Polygon, MultiPolygon
x = linspace(-4*pi,4*pi,100)

##### ENDPOINTS
# point A and B one
pA = array([0,0,0])
pB = array([20,20,20])

##### LINES
# line one
y = 2*x + 1
z = cos(2*x)
# line two
y = 2*sin(x) + sin(4*x) + 3*cos(x) + sin(2*x) + sin(x**2) - cos(6*x)
z = sin(x) + cos(3*x) -sin(x**2)
# line three
y = sin(x) + 0.8*sin(4*x) + cos(x) + 0.6*sin(2*x) + sin(x**2) - 1.2*cos(6*x)
z = sin(x) + 0.4*sin(4*x) - cos(x) - 0.6*sin(2*x) + sin(x**2) + 1.2*cos(6*x)

y = np.linspace(0,1.1,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
z = np.sin(np.cos(np.sin(x))) * np.sin(np.cos(x)) + np.sin(1)
line = np.array([[xc,yc,zc] for xc,yc,zc in zip(x,y,z)])
# scale line:
d = 10
line *= d

##### BARRIERS
building = Polygon([(0,10),
                    (5,10),
                    (5,5),
                    (0,5)
                    ])

military = Polygon([(15,10),
                    (25,10),
                    (25,0)
                    ])

nofly = Polygon([(0,25),
                (15,25),
                (15,20),
                (0,20)
                ])

test_hull_pts = np.array([[25,25,0],
                        [15,25,0],
                        [25,15,0],
                        [15,15,0],
                        [25,25,10],
                        [15,25,10],
                        [25,15,10],
                        [15,15,10]
                        ])


# Baby Steps..:
# HULLS PTS
pointset1 = np.array([[0,100,10],[50,100,10],[50,50,10],[0,50,10],
                    [0,100,0],[50,100,0],[50,50,0],[0,50,0]])

pointset2 = np.array([[60,30,30],[100,40,30],[100,0,30],[50,0,30],
                    [60,30,0],[100,40,0],[100,0,0],[50,0,0]])

int_line = np.array([[20,80,10],[80,20,10]])

above_line = np.array([[20,80,20],[80,20,50]])

nonint_line = np.array([[10,0,0],[100,90,100]])