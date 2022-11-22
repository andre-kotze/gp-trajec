# Validation functions for NSGA-III trajectories
import numpy as np

# for 2D implementation/testing we use shapely
from shapely.geometry import LineString, shape
from scipy.spatial import ConvexHull, Delaunay
from data.test_data_2d import barriers, pts # dict with multipolygons as values
#from data.test_data_3d import barriers_3d, pts_3d # dict with list of features (dicts) as values
from gptrajec import transform_2d, eaTrajec


def validate_2d(individual, params):
    # check intersection
    for barrier in barriers[params['barriers']].geoms:
        if individual.intersects(barrier):
            return False
    return True

def flexible_validate_2d(individual, params):
    intersection = 0
    for barrier in barriers[params['barriers']].geoms:
        intersection += barrier.intersection(individual).length
    return eval(params['int_cost'], {}, {"intersection": intersection})

#
#   2.5D (shapely)
#

def within(x, lower, upper):
    # this is the right way:
    return x >= lower and x <= upper
    # but now, all barriers are clamped to ground:
def too_low(x, upper):
    # x = iterable of z-coords
    return any(x <= upper)

def validate_2_5d(individual, params):
    # CHECK INTERSECTION
    # iterate through barriers dataset (feature dicts)
    for barrier in barriers_3d[params['barriers']]:
        # check solution intersection with feature geometry
        if individual.intersects(shape(barrier['geometry'])):
            # to intersect in 3D, must also lie "inside" barrier
            intersection = individual.intersection(shape(barrier['geometry']))
            if too_low([coord[2] for coord in intersection.coords], barrier['properties']['upper']):
                return False

    return True

#def flexible_validate_3d(individual, params):
#    intersection = 0
#    for barrier in barriers[params['barriers']].geoms:
#        intersection += barrier.intersection(individual).length
#    return eval(params['int_cost'], {}, {"intersection": intersection})

#
#   3D (scipy.spatial)
#

def pnt_in_cvex_hull_1(hull, pnt):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(np.concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    return False

def in_poly_hull_multi(poly, points):
    hull = ConvexHull(poly)
    res = []
    for p in points:
        new_hull = ConvexHull(np.concatenate((poly, [p])))
        res.append(np.array_equal(new_hull.vertices, hull.vertices))
    return res

def validate_3d(individual, params):
    val_type = params['validation_3d']
    # CHECK CONTAINMENT
    # iterate through barriers dataset (feature dicts)
    for barrier in barriers_3d[params['barriers']]:
        # check solution containment in feature geometry
        # ToDo: check if bounding boxes intersect. If not, early exit
        if val_type == 'hulls_equal1':
            if pnt_in_cvex_hull_1(poly, point):
                pass
        elif val_type == 'delaunay':
            if Delaunay(poly).find_simplex(point) >= 0:
                pass
        elif val_type == 'shapely':
            return validate_2_5d(individual, params)
        else:
            raise ValueError(f'Unrecognised 3D validation method {val_type}')

    return True