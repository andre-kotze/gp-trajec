import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
from gptrajec import transform_3d
from copy import deepcopy

y = np.linspace(0,1,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
z = np.sin(np.cos(np.sin(x))) * np.sin(np.cos(x)) + np.sin(1)
line = np.array([[xc,yc,zc] for xc,yc,zc in zip(x,y,z)])
print(line[0], line[-1])
test_line = deepcopy(line)
# normalise:
line = line / np.linalg.norm(line[-1])
print(f'normalised: {np.isclose(1, norm(line[-1]))} ({norm(line[-1])=})')
print(line[0], line[-1])
print(f'{np.linalg.norm(line[-1])=}')

#a = np.array([24,34,17])
#b = np.array([14,29,16])
#a = a / np.linalg.norm(a)
#b = b / np.linalg.norm(b)

dest_int = np.array([[288_000,48_600,10],[292_600,48_960,80]])
p, q = dest_int
dest_int_tf = dest_int - np.array([[288_000,48_600,10],[288_000,48_600,10]])
b = dest_int_tf[-1] # b is desired endpoint
# TRANSLATION TO ORIGIN:
# find deviance from ORIGIN
line[:,0] -= line[0,0]
line[:,1] -= line[0,1]
line[:,2] -= line[0,2]
print(line[0], line[-1])
a = line[-1] # a is actual endpoint

# normalise vectors
print(a,b)
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)
print('normalised vectors:\n', a,b)
# normalise line
#line = line / np.linalg.norm(line)

rot = R.align_vectors(np.reshape(b, (1, -1)),
                    np.reshape(a, (1, -1)))
#rot[0].as_matrix()

print(line.shape)
line_roted = rot[0].apply(line)
a2 = rot[0].apply(a)
print(line_roted.shape)
print(line_roted[0], line_roted[-1])

line3 = line.dot(rot[0].as_matrix())

# quick scale
d = np.linalg.norm(b) / np.linalg.norm(line_roted[-1])
print(f'scaling with {d=}')
line_roted *= d

print(f'rotated line from endpoint {line[-1]} to {line_roted[-1]}\nb is at {b}\n{np.rad2deg(rot[0].magnitude())=}')

fig = plt.figure()
ax0 = fig.add_subplot(231, projection='3d')
ax1 = fig.add_subplot(232, projection='3d')
ax2 = fig.add_subplot(233, projection='3d')
ax3 = fig.add_subplot(234, projection='3d')
ax4 = fig.add_subplot(235, projection='3d')
ax5 = fig.add_subplot(236, projection='3d')
ax0.plot([0,a[0]],[0,a[1]], [0,a[2]], color='g')
ax1.plot([0,a2[0]],[0,a2[1]], [0,a2[2]], color='r')
ax0.plot([0,b[0]],[0,b[1]], [0,b[2]], color='r')

ax2.plot(line_roted[:,0],line_roted[:,1], line_roted[:,2], color='b')
ax2.scatter(0,0,0, color='k')
ax2.scatter(b[0],b[1],b[2], color='k')
ax2.scatter(a[0],a[1],a[2], color='k')
ax2.plot(line[:,0],line[:,1], line[:,2], color='y')

ax3.plot(x,y,z, color='b')
#ax4.plot([0,a2[0]],[0,a2[1]], [0,a2[2]], color='r')
ax5.plot(line3[:,0],line3[:,1], line3[:,2], color='b')
for ax in [ax0,ax1,ax2,ax3, ax5]:
    ax.set_aspect('equal')

#test, ints = transform_3d(test_line, dest_int, intermediates=True)
test = transform_3d(test_line, dest_int)
print(f'{type(test)=}\n{np.round(test[:5],0)}')
ax4.plot(test[:,0],test[:,1], test[:,2], color='r')
ax4.scatter(p[0],p[1],p[2], color='k')
ax4.scatter(q[0],q[1],q[2], color='k')
#ax4.scatter(0,0,0, color='k')
print(f'{p=}\n{q=}')

plt.show()