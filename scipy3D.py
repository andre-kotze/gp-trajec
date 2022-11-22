import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from data.test_data_3d import line, test_hull_pts
from numpy import zeros, ones, arange, asarray, concatenate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

line[:,0] += 30
line[:,2] -= 5
line[:,1] += 10

def pnt_in_cvex_hull_1(hull, pnt):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices): 
        return True
    return False


def pnt_in_cvex_hull_2(hull_points, pnt):
    '''
    Given a set of points that defines a convex hull, uses simplex LP to determine
    whether point lies within hull.
    `hull_points` -- (N, 3) array of points defining the hull
    `pnt` -- point array of shape (3,)
    '''
    N = hull_points.shape[0]
    c = ones(N)
    A_eq = concatenate((hull_points, ones((N,1))), 1).T   # rows are x, y, z, 1
    b_eq = concatenate((pnt, (1,)))
    result = linprog(c, A_eq=A_eq, b_eq=b_eq)
    if result.success and c.dot(result.x) == 1.:
        return True
    return False

print('checking containments...\nverts in hull:')
cross = []
for n, vert in enumerate(line):
    if pnt_in_cvex_hull_1(ConvexHull(test_hull_pts), vert):
    #if pnt_in_cvex_hull_2(test_hull_pts, vert):
        print(n, end=' ', flush=True)
        cross.append(list(vert))
print()
cross = np.array(cross)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_hull_pts[:,0],test_hull_pts[:,1],test_hull_pts[:,2], color='r')
#ax.plot(test_hull_pts[:,0],test_hull_pts[:,1],test_hull_pts[:,2], color='r')
ax.plot(line[:,0],line[:,1],line[:,2], color='g')
ax.plot(cross[:,0],cross[:,1],cross[:,2], color='k')
ax.set_aspect('equal')
plt.show()