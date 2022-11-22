import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R

a = np.array([24,34,17])
b = np.array([17,24,34])

'''
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)
v = np.cross(a, b)
'''

'''
a2 = np.reshape(a, (1, -1))
b2 = np.reshape(b, (1, -1))

rot = R.align_vectors(a2,b2)
rot_mat = rot[0].as_matrix()

v = rot_mat.dot(b)
# b is mapped onto a

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([0,a[0]],[0,a[1]], [0,a[2]], color='r', alpha=0.5)
ax.plot([0,b[0]],[0,b[1]], [0,b[2]], color='g', alpha=0.5)
ax.plot([0,v[0]],[0,v[1]], [0,v[2]], color='b', alpha=0.5)

plt.show()

print('as quat: ', rot[0].as_quat())
'''
y = np.linspace(0,1.1,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
z = np.sin(np.cos(np.sin(x))) * np.sin(np.cos(x)) + np.sin(1)
line = np.array([[xc,yc,zc] for xc,yc,zc in zip(x,y,z)])
print(line[0], line[-1])
line = line / np.linalg.norm(line[-1])
print(line[0], line[-1])

#a = np.array([24,34,17])
#b = np.array([14,29,16])
#a = a / np.linalg.norm(a)
#b = b / np.linalg.norm(b)

dest_int = np.array([[288_000,48_600,10],[292_600,48_960,800]])
dest_int_tf = dest_int - np.array([[288_000,48_600,10],[288_000,48_600,10]])
b = dest_int_tf[-1]
# TRANSLATION TO ORIGIN:
# find deviance from ORIGIN
line[:,0] -= line[0,0]
line[:,1] -= line[0,1]
line[:,2] -= line[0,2]
print(line[0], line[-1])
a = line[-1]

# normalise vectors
print(a,b)
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)
print(a,b)
# normalise line
#line = line / np.linalg.norm(line)

rot = R.align_vectors(np.reshape(b, (1, -1)),
                    np.reshape(a, (1, -1)))
#rot[0].as_matrix()

print(line.shape)
line2 = rot[0].apply(line)
a2 = rot[0].apply(a)
print(line2.shape)
print(line2[0], line2[-1])

line3 = line.dot(rot[0].as_matrix())

# quick scale
d = np.linalg.norm(b) / np.linalg.norm(line2[-1])
line2 *= d

print(f'rotated line from endpoint {line[-1]} to {line2[-1]}\nb is at {b}\n{np.rad2deg(rot[0].magnitude())=}')

fig = plt.figure()
ax0 = fig.add_subplot(231, projection='3d')
ax1 = fig.add_subplot(232, projection='3d')
ax2 = fig.add_subplot(233, projection='3d')
ax3 = fig.add_subplot(234, projection='3d')
ax4 = fig.add_subplot(235, projection='3d')
ax5 = fig.add_subplot(236, projection='3d')
ax0.plot([0,a[0]],[0,a[1]], [0,a[2]], color='g')
ax0.plot([0,a2[0]],[0,a2[1]], [0,a2[2]], color='r')
ax0.plot([0,b[0]],[0,b[1]], [0,b[2]], color='r')
ax2.plot(line2[:,0],line2[:,1], line2[:,2], color='b')
ax2.scatter(0,0,0, color='k')
ax2.scatter(b[0],b[1],b[2], color='k')
ax3.plot(x,y,z, color='b')
ax2.plot(line[:,0],line[:,1], line[:,2], color='y')
ax4.plot([0,a2[0]],[0,a2[1]], [0,a2[2]], color='r')
ax5.plot(line3[:,0],line3[:,1], line3[:,2], color='b')
for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
    ax.set_aspect('equal')
plt.show()