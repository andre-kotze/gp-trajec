import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, shape, mapping
import fiona

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

barrier_set = MultiPolygon([building, military, nofly])
clokes = []
with fiona.open('./data/clove_lakes.gpkg', layer='clove_lakes') as layer:
    for feat in layer:
        clokes.append(shape(feat['geometry']))
clokes = [cloke[0] for cloke in clokes]

islas = []
with fiona.open('./data/mediterranean.gpkg', layer='mediterranean') as layer:
    for feat in layer:
        islas.append(shape(feat['geometry']))
islas = [isla[0] for isla in islas]

#journey = (Point(-74.1412,40.6053), Point(-74.05937,40.6377))
journey = (Point(-74.1412,40.6200), Point(-74.05937,40.6377))
#journey = (Point(5547780,1980024), Point(5727123,1564999))