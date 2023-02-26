import numpy as np
from shapely.geometry import MultiLineString, Polygon, MultiPolygon, Point, shape
from gptrajec import hull_from_poly
import fiona

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

staten = []
with fiona.open('./data/clove_lakes_z.gpkg', layer='clove_lakes_z') as layer:
    for feat in layer:
        staten.append(shape(feat['geometry']).geoms[0])
staten = MultiPolygon(staten)

staten_simp = []
with fiona.open('./data/clove_lakes_z.gpkg', layer='simplified') as layer:
    for feat in layer:
        staten_simp.append(shape(feat['geometry']).geoms[0])
staten_simp = MultiPolygon(staten_simp)

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

exampspc_simp = []
#with fiona.open('./data/exampspc_simp_z_added.gpkg', layer='z_added') as layer:
with fiona.open('./data/example_space_subset.gpkg', layer='simp_z') as layer:
    for feat in layer:
        exampspc_simp.append(shape(feat['geometry']).geoms[0])
exampspc_simp = MultiPolygon(exampspc_simp)

example_hulls = [hull_from_poly(poly) for poly in example_space.geoms]
cl_simp_hulls = [hull_from_poly(poly) for poly in clokes_simp.geoms]
cl_simp_polies = [hull_from_poly(poly, True) for poly in clokes_simp.geoms]
example_points = [hull.points for hull in example_hulls]

barriers3 = {'clokes': clokes,
            'clokes_simp': clokes_simp,
            'cl_simp_hulls': cl_simp_hulls,
            'cl_simp_polies': cl_simp_polies,
            'example_space': example_space,
            'example_space_simp': exampspc_simp,
            'staten': staten,
            'staten_simp': staten_simp,}

pts3['ex0'] = Point(list(example_route.geoms[0].coords[0]))
pts3['ex1'] = Point(list(example_route.geoms[0].coords[-1]))
pts3['nex0'] = Point(319641.7,59339.5,20)
pts3['triv'] = Point(320660.8,60105.7,20)