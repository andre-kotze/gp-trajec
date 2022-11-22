import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def align_vectors(a, b):
     v = np.cross(a, b)
     s = np.linalg.norm(v)
     c = np.dot(a, b)

     v1, v2, v3 = v
     h = 1 / (1 + c)

     Vmat = np.array([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]])

     R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
     return R

def angle(a, b):
    """Angle between vectors"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.arccos(a.dot(b))

point = np.array([-0.2, 1.1, -0.2])
direction = np.array([1., 0., 0.])
rotation = align_vectors(point, direction)

# Rotate point in align with direction. The result vector is aligned with direction
result = rotation.dot(point)
print(result)
print('Angle:', angle(direction, point)) # 0.0
print('Length:', np.isclose(np.linalg.norm(point), np.linalg.norm(result))) # True


# Rotate direction by the matrix, result does not align with direction but the 
# angle between the original vector (direction) and the result2 are the same.
result2 = rotation.dot(direction)
print(result2)
print('Same Angle:', np.isclose(angle(point,result), angle(direction,result2))) # True
print('Length:', np.isclose(np.linalg.norm(direction), np.linalg.norm(result2))) # True


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0,point[0]],[0,point[1]], [0,point[2]], color='y')
ax.plot([0,direction[0]],[0,direction[1]], [0,direction[2]], color='c')
ax.plot([0,result[0]],[0,result[1]], [0,result[2]], color='r')
ax.plot([0,result2[0]],[0,result2[1]], [0,result2[2]], color='g')
plt.show()