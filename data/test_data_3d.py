import numpy as np
from numpy import sin, cos, arctan2, pi, linspace, array
from shapely.geometry import MultiLineString, Polygon, MultiPolygon, Point, shape
from gptrajec import hull_from_poly
import fiona

##### BARRIERS
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
with fiona.open('./data/clove_lakes_z.gpkg', layer='clove_lakes_z') as layer:
    for feat in layer:
        clokes.append(shape(feat['geometry']).geoms[0])
clokes = MultiPolygon(clokes)

clokes_simp = []
with fiona.open('./data/clove_lakes_z.gpkg', layer='simplified') as layer:
    for feat in layer:
        clokes_simp.append(shape(feat['geometry']).geoms[0])
clokes_simp = MultiPolygon(clokes_simp)

journeys = {
    'bl': Point(288501.2,48648.0,5), 
    'tr': Point(294681.4,53744.4,5),
    'tl': Point(294952.3,49073.8,5), 
    'br': Point(287778.7,53150.9,5),
    'tc': Point(291107.6,53789.1,50), 
    'bc': Point(292653.3,48962.3,50),
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

example_hulls = [hull_from_poly(poly) for poly in example_space.geoms]
cl_simp_hulls = [hull_from_poly(poly) for poly in clokes_simp.geoms]
example_points = [hull.points for hull in example_hulls]

barriers3 = {'clokes': clokes,
            'clokes_simp': clokes_simp,
            'cl_simp_hulls': cl_simp_hulls,
            'example_space': example_space}

pts3['ex0'] = Point(list(example_route.geoms[0].coords[0]))
pts3['ex1'] = Point(list(example_route.geoms[0].coords[-1]))
pts3['nex0'] = Point(319641.7,59339.5,20)
pts3['triv'] = Point(320660.8,60105.7,20)