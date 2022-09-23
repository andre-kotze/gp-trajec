import numpy as np
#from math import atan2
import matplotlib.pyplot as plt


def transform_2d(line, dest):
    '''
    Takes a line and transforms it to
    map onto the interval dest, i.e. 
    between a start and end point
    line: list of coordinate list pairs
    dest: list of coordinate list pairs of length 2
    '''
    # Endpoints:
    ((ax, ay), (bx, by)) = dest
    (px, py), (qx, qy) = line[0], line[-1]

    # TRANSLATION:
    # find deviance from point a
    dx = px - ax
    dy = py - ay
    print(f'Translation: dx = {dx}, dy = {dy}')
    line[:,0] -= dx
    line[:,1] -= dy
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    print(f'Scale factor: {scale_factor}\nLine: {line_dist}\nDist: {dest_dist}')
    scale_matrix = np.array([[scale_factor,0,0],
                            [0,scale_factor,0],
                            [0,0,1]])
    print(scale_matrix)
    # ROTATION:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    print(f'Delta theta: {np.rad2deg(theta)}°')
    c = np.cos(theta)
    d = np.sin(theta)
    rotation_matrix = np.array([[c,-d,0],
                                [d,c,0],
                                [0,0,1]])
    print(rotation_matrix)
    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], np.ones(len(line)))])
    print(transform_matrix)
    return line

def transform_3d(line, dest):
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
    print(f'Translation: dx = {dx}, dy = {dy}, dz = {dz}')
    line[:,0] -= dx
    line[:,1] -= dy
    line[:,2] -= dz
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
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
    print(transform_matrix)
    return line



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