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
with fiona.open('./data/clove_lakes_utm.gpkg', layer='clove_lakes_utm') as layer:
    for feat in layer:
        clokes.append(shape(feat['geometry']).geoms[0])
clokes = MultiPolygon(clokes)

islas = []
with fiona.open('./data/mediterranean.gpkg', layer='mediterranean') as layer:
    for feat in layer:
        islas.append(shape(feat['geometry']).geoms[0])
islas = MultiPolygon(islas)

journeys = {
    'bl': Point(288501.2,48648.0), 
    'tr': Point(294681.4,53744.4),
    'tl': Point(294952.3,49073.8), 
    'br': Point(287778.7,53150.9),
    'bc': Point(291107.6,53789.1), 
    'tc': Point(292653.3,48962.3),
    'grc00': Point(5680405,1889944), 
    'grc01': Point(5579206,1609699),
    'grc10': Point(5533696,2073182), 
    'grc11': Point(5579206,1609699),
    'lc': Point(5533696,2073182), 
    'rc': Point(5579206,1609699)
}
pts = {}
with fiona.open('./data/clove_lakes_pts.gpkg', layer='pts') as layer:
    for feat in layer:
        pts[feat['properties']['name']] = shape(feat['geometry'])
pts.update(journeys)

example_route = []
with fiona.open('./data/example_route.gpkg', layer='example_route') as layer:
    for feat in layer:
        example_route.append(shape(feat['geometry']))
example_route = example_route[0]

example_space = []
with fiona.open('./data/example_space.gpkg', layer='example_space') as layer:
    for feat in layer:
        example_space.append(shape(feat['geometry']).geoms[0])
example_space = MultiPolygon(example_space)

barriers = {'clokes': clokes,
            'islas': islas,
            'example_space': example_space}