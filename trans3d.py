import numpy as np

def get_rotation_matrix(A,B):
# a and b are in the form of numpy array

   ax = A[0]
   ay = A[1]
   az = A[2]

   bx = B[0]
   by = B[1]
   bz = B[2]

   au = A/(np.sqrt(ax*ax + ay*ay + az*az))
   bu = B/(np.sqrt(bx*bx + by*by + bz*bz))

   R=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])


   return(R)

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

    # TRANSLATION TO ORIGIN:
    # find deviance from ORIGIN
    line[:,0] -= px
    line[:,1] -= py
    line[:,2] -= pz
    # now the line originates at the origin
    
    # SCALING:
    line_dist = np.linalg.norm(line[-1] - line[0])
    dest_dist = np.linalg.norm(dest[-1] - dest[0])
    scale_factor = dest_dist / line_dist
    if printing:
        print(f'Scale factor: {scale_factor:.2f}\nLine: {line_dist:.2f}\nDist: {dest_dist:.2f}')
    scale_matrix = np.array([[scale_factor,0,0,0],
                            [0,scale_factor,0,0],
                            [0,0,scale_factor,0],
                            [0,0,0,1]])
    #scale_matrix = np.array([[scale_factor,0,0],
    #                        [0,scale_factor,0],
    #                        [0,0,scale_factor]])

    post_scale = np.array([
        np.array(coord_set).dot(scale_matrix) 
        for coord_set in zip(line[:,0], line[:,1], line[:,2], np.ones(100))])
    #post_scale = post_scale[:,0:3]


    # ROTATION:
    # = = = = = about z:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    if printing:
        #print(f'About z-axis: {line_angle=}, {dest_angle=}')
        print(f'Delta theta about z-axis: {np.rad2deg(theta):.2f}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_z = np.array([[c,-s,0,0],
                        [s,c,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])

    #rot_mat_z = np.array([[c,-s,0],
    #                    [s,c,0],
    #                    [0,0,1]])

    post_z = post_scale.dot(rot_mat_z)
    # transform the endpoint, before the next rotation:
    #(qx, qy, qz, _) = np.array([qx, qy, qz, 1]).dot(rot_mat_z)
    (qx, qy, qz, _) = post_z[-1]



    # = = = = = about y:
    line_angle = np.arctan2(qx - px, qz - pz)
    dest_angle = np.arctan2(bx - ax, bz - az)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        #print(f'About y-axis: {line_angle=}, {dest_angle=}')
        print(f'Delta theta about y-axis: {np.rad2deg(theta):.2f}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_y = np.array([[c,0,s,0],
                        [0,1,0,0],
                        [-s,0,c,0],
                        [0,0,0,1]])
    #rot_mat_y = np.array([[c,0,s],
    #                    [0,1,0],
    #                    [-s,0,c]])


    post_y = post_z.dot(rot_mat_y)
    # transform the endpoint, before the next rotation:
    #(qx, qy, qz, _) = np.array([qx, qy, qz, 1]).dot(rot_mat_y)
    (qx, qy, qz, _) = post_y[-1]

    '''
    # = = = = = about x:
    line_angle = np.arctan2(qz - pz, qy - py)
    dest_angle = np.arctan2(bz - az, by - ay)
    theta = dest_angle - line_angle
    theta *= -1
    if printing:
        #print(f'About x-axis: {line_angle=}, {dest_angle=}')
        print(f'Delta theta about x-axis: {np.rad2deg(theta):.2f}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_x = np.array([[1,0,0,0],
                        [0,c,-s,0],
                        [0,s,c,0],
                        [0,0,0,1]])
    #rot_mat_x = np.array([[1,0,0],
    #                    [0,c,-s],
    #                    [0,s,c]])




    post_x = post_y.dot(rot_mat_x)
    #(qx, qy, qz, _) = post_x[-1]
    '''
    # NEW ABT Z AGAIN:
    # = = = = = about z:
    line_angle = np.arctan2(qy - py, qx - px)
    dest_angle = np.arctan2(by - ay, bx - ax)
    theta = dest_angle - line_angle
    # negate, to rotate c-clockwise
    theta *= -1
    if printing:
        #print(f'About z-axis: {line_angle=}, {dest_angle=}')
        print(f'Delta theta about z-axis: {np.rad2deg(theta):.2f}°')
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat_z2 = np.array([[c,-s,0,0],
                        [s,c,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])

    #rot_mat_z = np.array([[c,-s,0],
    #                    [s,c,0],
    #                    [0,0,1]])

    post_z2 = post_y.dot(rot_mat_z2)
    # transform the endpoint, before the next rotation:
    #(qx, qy, qz, _) = np.array([qx, qy, qz, 1]).dot(rot_mat_z)
    (qx, qy, qz, _) = post_z2[-1]


    #############################################




    rotation_matrix = rot_mat_z @ rot_mat_y @ rot_mat_z2
    print('### ROT MAT ###\n', rotation_matrix)


    # TRANSFORMATION:
    transform_matrix = scale_matrix @ rotation_matrix
    line = np.array([np.array(coord_set).dot(transform_matrix) for coord_set in zip(line[:,0], line[:,1], line[:,2], np.ones(100))])
    
    # TRANSLATE TO DEST
    # find deviance from Point a
    dx = line[0,0] - ax
    dy = line[0,1] - ay
    dz = line[0,2] - az

    line[:,0] -= dx
    line[:,1] -= dy
    line[:,2] -= dz

    if printing:
        print(rotation_matrix)
    return line[:,0:3], [post_scale, post_z, post_y, post_z2]
    # numpy array like [[x,y,z,1],[x,y,z,1],...]
