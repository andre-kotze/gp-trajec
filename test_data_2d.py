import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon
x = 0

##### ENDPOINTS
# point A and B one
pA = np.array([0,0])
pB = np.array([20,20])
a, b = [pA[0],pB[0]],[pA[1],pB[1]]


##### LINES
# line one
y = 2*x + 1
# line two
y = 2*np.sin(x) + np.sin(4*x) + 3*np.cos(x) + np.sin(2*x) + np.sin(x**2) - np.cos(6*x)
# line three
y = np.sin(x) + 0.8*np.sin(4*x) + np.cos(x) + 0.6*np.sin(2*x) + np.sin(x**2) - 1.2*np.cos(6*x)


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