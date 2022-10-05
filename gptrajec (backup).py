import numpy as np
#from math import atan2
#import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def transform_2d(line, dest, printing=False):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: numpy array of coordinate list pairs
    dest: list of coordinate list pairs of length 2
    '''
    # Endpoints:
    ((ax, ay), (bx, by)) = dest
    (px, py), (qx, qy) = line[0], line[-1]

    # TRANSLATION:
    # find deviance from point a
    dx = px - ax
    dy = py - ay
    if printing:
        print(f'Translation: dx = {dx}, dy = {dy}')

    line[:,0] -= dx
    line[:,1] -= dy
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0],
                            [0,scale_factor,0],
                            [0,0,1]])
    if printing:
        print(scale_matrix)
    scaled = np.array([np.array(coord_set).dot(scale_matrix) for coord_set in zip(line[:,0], line[:,1], np.zeros(len(line)))])
    # ROTATION:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    if printing:
        print(f'Delta theta: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    d = np.sin(theta)
    rotation_matrix = np.array([[c,-d,0],
                                [d,c,0],
                                [0,0,1]])
    a = ax
    b = ay
    prerotation_translation_matrix = np.array([[1,0,-a],
                                                [0,1,-b],
                                                [0,0,1]])
    postrotation_translation_matrix = np.array([[1,0,a],
                                                [0,1,b],
                                                [0,0,1]])
    if printing:
        print(f'prerotation: \n{prerotation_translation_matrix}')
        print(f'rotation: \n{rotation_matrix}')
        print(f'postrotation: \n{postrotation_translation_matrix}')

    prerotation = scaled.dot(prerotation_translation_matrix)
    rotation = prerotation.dot(rotation_matrix)
    postrotation = rotation.dot(postrotation_translation_matrix)
    if printing:
        print(scaled[:5])
        print(prerotation[:5])
        print(rotation[:5])
        print(postrotation[:5])
    output = postrotation
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.ones(len(line)))])
    #if printing:
    #    print(transform_matrix)
    return line # numpy array like [[x,y,1],[x,y,1]...]

def transform_3d(line, dest, printing=False):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: list of coordinate list triplets
    dest: list of coordinate list triplets of length 2
    '''
    # Endpoints:
    ((ax, ay, az), (bx, by, bz)) = dest
    (px, py, pz), (qx, qy, qz) = line[0], line[-1]

    # TRANSLATION:
    # find deviance from point a
    dx = px - ax
    dy = py - ay
    dz = pz - az
    if printing:
        print(f'Translation: dx = {dx}, dy = {dy}, dz = {dz}')
    line[:,0] -= dx
    line[:,1] -= dy
    line[:,2] -= dz
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0,0],
                            [0,scale_factor,0,0],
                            [0,0,scale_factor,0],
                            [0,0,0,1]])
    # ROTATION:
    # = = = = = about z:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about z-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_z = np.array([[c,-s,0,0],
                        [s,c,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    # = = = = = about y:
    line_angle = np.arctan2(qx - px, qz - pz)
    dest_angle = np.arctan2(bx - ax, bz - az)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about y-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_y = np.array([[c,0,-s,0],
                        [0,1,0,0],
                        [s,0,c,0],
                        [0,0,0,1]])
    # = = = = = about x:
    line_angle = np.arctan2(qz - pz, qy - py)
    dest_angle = np.arctan2(bz - az, by - ay)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        print(f'Delta theta about x-axis: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_x = np.array([[1,0,0,0],
                        [0,c,s,0],
                        [0,-s,c,0],
                        [0,0,0,1]])
    rotation_matrix = rot_mat_z @ rot_mat_y @ rot_mat_x
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.ones(len(line)))])
    if printing:
        print(transform_matrix)
    return line

def faces_from_poly(polygon):
    # ToDo
    # check not empty
    # check 3D
    # init list;
    polygons = [polygon]
    coords = polygon.exterior.coords
    for pt in range(len(coords) - 1):
        poly = [coords[pt], coords[pt + 1]]
        polybase = [(tpl[0],tpl[1],0) for tpl in poly]
        poly.extend(polybase.reverse())
        polygons.append(Polygon(poly))

    return polygons

# move this somewhere else....
def read_geodata(geodata, h, base=0):
    # geodata: shp, gpkg, geojson, kml, csv of zones
    # h: field to use for barrier height
    # base: base/minimum altitude. Defaults to ground
    #   but a field can be specified

    barrier_meshes = []
    return barrier_meshes
'''
def __main__():
    x = np.linspace(-5,5,100)
    y = 2*np.sin(x) + np.sin(4*x) + 3*np.cos(x) + np.sin(2*x) + np.sin(x**2) - np.cos(6*x)
    z = np.ones(100)
    pA = np.array([0,0])
    pB = np.array([20,20])
    a, b = [pA[0],pB[0]],[pA[1],pB[1]]
    line = np.column_stack((x,y))
    t_line = transform(line, [pA,pB])
    t_line = np.array(t_line)
    plt.plot(t_line[:,0],t_line[:,1])
    plt.plot(a,b)
    plt.show()

__main__()
'''