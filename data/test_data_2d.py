import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point, shape, mapping
import fiona

clokes = []
with fiona.open('./data/clove_lakes_utm.gpkg', layer='clove_lakes_utm') as layer:
    for feat in layer:
        clokes.append(shape(feat['geometry']).geoms[0])
clokes = MultiPolygon(clokes)

islas = []
with fiona.open('./data/mediterranean.gpkg', layer='fixed') as layer:
    for feat in layer:
        islas.append(shape(feat['geometry']).geoms[0])
islas = MultiPolygon(islas)

journeys = {
    'bl': Point(288501.2,48648.0), 
    'tr': Point(294681.4,53744.4),
    'tl': Point(294952.3,49073.8), 
    'br': Point(287778.7,53150.9),
    'tc': Point(291107.6,53789.1), 
    'bc': Point(292653.3,48962.3),
    #'grc00': Point(5680405,1889944), 
    #'grc01': Point(5579206,1609699),
    #'grc10': Point(5533696,2073182), 
    #'grc11': Point(5579206,1609699),
    #'lc': Point(5533696,2073182), 
    #'rc': Point(5579206,1609699)
}
pts = {}
with fiona.open('./data/clove_lakes_pts.gpkg', layer='pts') as layer:
    for feat in layer:
        pts[feat['properties']['name']] = shape(feat['geometry'])

with fiona.open('./data/porto.gpkg', layer='porto_pts') as layer:
    for feat in layer:
        pts[feat['properties']['name']] = shape(feat['geometry'])

pts.update(journeys)

with fiona.open('./data/mediterranean.gpkg', layer='pts') as layer:
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

porto = []
#with fiona.open('./data/exampspc_simp_z_added.gpkg', layer='z_added') as layer:
with fiona.open('./data/porto.gpkg', layer='simplified') as layer:
    for feat in layer:
        porto.append(shape(feat['geometry']).geoms[0])
porto = MultiPolygon(porto)

barriers = {'clokes': clokes,
            'clokes_simp':clokes,
            'aegean': islas,
            'example_space': example_space,
            'porto': porto}

pts['ex0'] = Point(example_route.geoms[0].coords[0])
pts['ex1'] = Point(example_route.geoms[0].coords[-1])
pts['nex0'] = Point(319641.7,59339.5)
pts['triv'] = Point(320660.8,60105.7)