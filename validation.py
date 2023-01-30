# Validation functions for NSGA-III trajectories
import numpy as np

# for 2D implementation/testing we use shapely
from shapely.geometry import LineString, shape, MultiLineString
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import linprog
from data.test_data_2d import barriers, pts # dict with multipolygons as values
from data.test_data_3d import barriers3, pts3, example_hulls, example_points, global_max_z # dict with list of features (dicts) as values
from gptrajec import transform_2d, eaTrajec
import moeller_trumbore_algo as mt

def convex_hull(poly):
    high_pts = np.array(poly.exterior.coords)
    x, y = poly.exterior.xy
    z = np.zeros(len(x))
    low_pts = np.column_stack((x,y,z))
    pts = np.append(high_pts, low_pts, axis=0)
    return ConvexHull(pts)

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

def within(x, lower=0, upper=300):
    # this is the right way:
    return x >= lower and x <= upper
    # but now, all barriers are clamped to ground:
def too_low(z, upper):
    # z = iterable of z-coords
    return any(zc <= upper for zc in z)

def validate_2_5d(individual, params):
    # CHECK INTERSECTION
    # iterate through barriers dataset (feature dicts)
    for barrier in barriers3[params['barriers']].geoms:
        z = barrier.exterior.coords[0][2]
        #print(len(barrier.exterior.coords))
        # check solution intersection with feature geometry
        if individual.intersects(barrier):
            # to intersect in 3D, must also lie "inside" barrier
            intersection = individual.intersection(barrier)
            if isinstance(intersection, MultiLineString):
                for part in intersection.geoms:
                    if too_low([coord[2] for coord in part.coords],z):
                        return False
                    elif any([within(coord[2]) for coord in part.coords]):
                        return False
            else:
                if too_low([coord[2] for coord in intersection.coords],z):
                    return False
                elif any([within(coord[2]) for coord in intersection.coords]):
                    return False
    return True

def flexible_validate_2_5d(individual, params):
    intersection = 0
    for barrier in barriers3[params['barriers']].geoms:
        z = barrier.exterior.coords[0][2]
        if individual.intersects(barrier):
            intersect_geom = individual.intersection(barrier)
            if isinstance(intersect_geom, MultiLineString):
                for part in intersect_geom.geoms:
                    intersection += (part.length * (len([coord for coord in part.coords if coord[2] < z]) / len(part.coords)))
            else:
                intersection += (intersect_geom.length * (len([coord for coord in intersect_geom.coords if coord[2] < z]) / len(intersect_geom.coords)))
        clipd_verts = [coord for coord in individual.coords if coord[2] < 0 or coord[2] > global_max_z]
        intersection += (individual.length * (len(clipd_verts) / len(individual.coords)))
    #print(intersection, individual.length)
    return eval(params['int_cost'], {}, {"intersection": intersection})

#
#   3D (scipy.spatial)
#

def hulls_equal(hull, pnt):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(np.concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    return False

def hulls_equal_multi(poly, points):
    hull = ConvexHull(poly)
    res = []
    for p in points:
        new_hull = ConvexHull(np.concatenate((poly, [p])))
        res.append(np.array_equal(new_hull.vertices, hull.vertices))
    return res

def linprog_val(hull_points, pnt): # apparently quite slow !!
    '''
    Given a set of points that defines a convex hull, uses simplex LP to determine
    whether point lies within hull.
    `hull_points` -- (N, 3) array of points defining the hull
    `pnt` -- point array of shape (3,)
    '''
    N = hull_points.shape[0]
    c = np.ones(N)
    A_eq = np.concatenate((hull_points, np.ones((N,1))), 1).T   # rows are x, y, z, 1
    b_eq = np.concatenate((pnt, (1,)))
    result = linprog(c, A_eq=A_eq, b_eq=b_eq)
    if result.success and c.dot(result.x) == 1.:
        return True
    return False

def triangles(vertices, ray_origin, ray_direction):
    intersection = mt.ray_triangle_intersection(vertices, ray_origin, ray_direction)
    return intersection

def delauney_val(hulls, points):
    for hull in hulls:
        for point in points:
            if Delaunay(hull).find_simplex(point) >= 0: # True when intersecting
                return False
    return True

def validate_3d(individual, params):
    val_type = params['validation_3d']
    # CHECK CONTAINMENT
    # iterate through barriers dataset (feature dicts)
    #for barrier in barriers3[params['barriers']].geoms:
        # check solution containment in feature geometry
        # ToDo: check if bounding boxes intersect. If not, early exit
    if val_type == 'hulls_equal':
        if hulls_equal(poly, point):
            pass
    elif val_type == 'delaunay':
        return delauney_val(barriers3[params['barriers']], list(individual.coords))
    elif val_type == 'shapely':
        return validate_2_5d(individual, params)
    elif val_type == 'linprog':
        if linprog_val():
            pass
    elif val_type == 'triangles':
        if triangles():
            pass
    elif val_type == 'postgis':
        pass
    else:
        raise ValueError(f'Unrecognised 3D validation method {val_type}')

    return True