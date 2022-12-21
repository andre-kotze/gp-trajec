import numpy as np
from numpy import sin, cos, arctan2, pi, linspace, array
from shapely.geometry import MultiLineString, Polygon, MultiPolygon, Point, shape
from scipy.spatial import ConvexHull
import fiona
x = linspace(-4*pi,4*pi,100)

def convex_hull(poly):
    high_pts = np.array(poly.exterior.coords)
    x, y = poly.exterior.xy
    z = np.zeros(len(x))
    low_pts = np.column_stack((x,y,z))
    pts = np.append(high_pts, low_pts, axis=0)
    return ConvexHull(pts)

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

clokes = []
with fiona.open('./data/clove_lakes_z.gpkg', layer='simplified') as layer:
    for feat in layer:
        clokes.append(shape(feat['geometry']).geoms[0])
clokes = MultiPolygon(clokes)

islas = []
with fiona.open('./data/mediterranean.gpkg', layer='mediterranean') as layer:
    for feat in layer:
        islas.append(shape(feat['geometry']).geoms[0])
islas = MultiPolygon(islas)

journeys = {
    'bl': Point(288501.2,48648.0,5), 
    'tr': Point(294681.4,53744.4,5),
    'tl': Point(294952.3,49073.8,5), 
    'br': Point(287778.7,53150.9,5),
    'tc': Point(291107.6,53789.1,50), 
    'bc': Point(292653.3,48962.3,50),
    'grc00': Point(5680405,1889944), 
    'grc01': Point(5579206,1609699),
    'grc10': Point(5533696,2073182), 
    'grc11': Point(5579206,1609699),
    'lc': Point(5533696,2073182,5), 
    'rc': Point(5579206,1609699,5)
}
pts3 = {}
with fiona.open('./data/clove_lakes_pts.gpkg', layer='pts') as layer:
    for feat in layer:
        pts3[feat['properties']['name']] = shape(feat['geometry'])
pts3.update(journeys)

example_route = []
with fiona.open('./data/example_route_z.gpkg', layer='z_added') as layer:
    for feat in layer:
        example_route.append(shape(feat['geometry']))
example_route = MultiLineString(example_route)

example_space = []
#with fiona.open('./data/example_space_z_added.gpkg', layer='z_added') as layer:
with fiona.open('./data/example_space_subset.gpkg', layer='example_space_subset') as layer:
    for feat in layer:
        example_space.append(shape(feat['geometry']).geoms[0])
example_space = MultiPolygon(example_space)

example_hulls = [convex_hull(poly) for poly in example_space.geoms]
example_points = [hull.points for hull in example_hulls]

barriers3 = {'clokes': clokes,
            'islas': islas,
            'example_space': example_space}

pts3['ex0'] = Point(list(example_route.geoms[0].coords[0]))
pts3['ex1'] = Point(list(example_route.geoms[0].coords[-1]))
pts3['nex0'] = Point(319641.7,59339.5,20)
pts3['triv'] = Point(320660.8,60105.7,20)