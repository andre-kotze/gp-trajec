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