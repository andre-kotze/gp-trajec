from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from gptrajec import transform_3d
from trans3d import transform_3d as transform_zyz

plot_types = ['steps', 'party', 'result', 'flat']
plot_type = plot_types[3]

y = np.linspace(0,1,100)
#x = np.linspace(0,1,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
#y = 2*np.sin(x) + np.sin(4*x) + 3*np.cos(x) + np.sin(2*x) + np.sin(x**2) - np.cos(6*x)
z = np.sin(np.cos(np.sin(x))) * np.sin(np.cos(x)) + np.sin(1)
#z = -np.sin(1)*x - np.sin(np.cos(x)) * np.sin(x) -np.cos(1) - np.cos(np.cos(-1-x))
y = -np.sin(y**1.4)

fig = plt.figure()

dest_int = np.array([[288_000,48_600,10],[292_600,48_960,800]])
dest_int_tf = dest_int - np.array([[288_000,48_600,10],[288_000,48_600,10]])
line = np.array([[xc,yc,zc] for xc,yc,zc in zip(x,y,z)])
print(f'transforming route with {line.shape=}\nfrom {np.round(line[0],2)} to {np.round(line[-1],2)}')
#route, steps = transform_3d(line, dest_int, printing=True, intermediates=True)
route, steps = transform_zyz(line, dest_int, printing=True)
post_scale, post_z, post_y, post_z2 = steps
print(f'transformed route with {route.shape=}\nfrom {np.round(route[0],2)} to {np.round(route[-1],2)}')
length = np.linalg.norm(route[-1]-route[0])
print(f'{length:.2f}m straight line')
offset = np.linalg.norm(dest_int[-1] - route[-1])
print(f'\n\noffset from true endpoint: {offset:.2f}\n({np.round(dest_int[-1]-route[-1],2)})')
print(f'precision: {offset/length}')

#ax1.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')

if plot_type == plot_types[0]:
    ax0 = fig.add_subplot(231, projection='3d')
    ax0.set_title('Solution curve')
    ax1 = fig.add_subplot(232, projection='3d')
    ax1.set_title('Scaled')
    ax2 = fig.add_subplot(233, projection='3d')
    ax2.set_title('Rotated z')
    ax3 = fig.add_subplot(234, projection='3d')
    ax3.set_title('Rotated y')
    ax4 = fig.add_subplot(235, projection='3d')
    ax4.set_title('Rotated z2')
    ax5 = fig.add_subplot(236, projection='3d')
    ax5.set_title('Final route')
    ax0.plot(x,y,z)
    ax1.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2])
    ax2.plot(post_z[:,0], post_z[:,1], post_z[:,2])
    ax3.plot(post_y[:,0], post_y[:,1], post_y[:,2])
    ax4.plot(post_z2[:,0], post_z2[:,1], post_z2[:,2])
    ax5.plot(route[:,0], route[:,1], route[:,2])
    ax5.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
    for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
        ax.set_aspect('equal')
elif plot_type == plot_types[1]: # party
    ax2 = fig.add_subplot(121, projection='3d')
    ax2.set_title('Rotation (z,y,x)')
    ax3 = fig.add_subplot(122, projection='3d')
    ax3.set_title('Final route')
    ax2.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2], color='y')
    ax2.plot(post_z[:,0], post_z[:,1], post_z[:,2], color='r')
    ax2.plot(post_y[:,0], post_y[:,1], post_y[:,2], color='g')
    ax2.plot(post_z2[:,0], post_z2[:,1], post_z2[:,2], color='b')
    ax3.plot(route[:,0], route[:,1], route[:,2])
    ax3.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='m')
    for ax in [ax2,ax3]:
        ax.set_aspect('equal')
elif plot_type == plot_types[2]:
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')
    ax0.plot(x,y,z)
    ax1.plot(route[:,0], route[:,1], route[:,2])
elif plot_type == plot_types[3]: # flat
    ax0 = fig.add_subplot(231, projection='3d')
    ax0.set_title('Scaled solution curve')
    ax1 = fig.add_subplot(232)
    ax1.set_title('Rotation in z')
    ax2 = fig.add_subplot(233)
    ax2.set_title('Rotation in y')
    ax3 = fig.add_subplot(234)
    ax3.set_title('Rotation in z2')
    ax4 = fig.add_subplot(235, projection='3d')
    ax4.set_title('Steps')
    ax5 = fig.add_subplot(236, projection='3d')
    ax5.set_title('Final route')

    ax0.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2], color='y')
    ax0.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='r')
    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')

    ax1.plot(post_scale[:,0], post_scale[:,1], color='y') # x,y
    ax1.plot(post_z[:,0], post_z[:,1], color='r')# x,y
    ax1.scatter(dest_int_tf[:,0], dest_int_tf[:,1], color='b')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.plot(post_z[:,0], post_z[:,2], color='r')# x,z
    ax2.plot(post_y[:,0], post_y[:,2], color='g')# x,z
    ax2.scatter(dest_int_tf[:,0], dest_int_tf[:,2], color='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')

    ax3.plot(post_y[:,0], post_y[:,1], color='g')# x,y
    ax3.plot(post_z2[:,0], post_z2[:,1], color='b')# x,y
    ax3.scatter(dest_int_tf[:,0], dest_int_tf[:,1], color='b')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    ax4.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2], color='y')
    ax4.plot(post_z[:,0], post_z[:,1], post_z[:,2], color='r')
    ax4.plot(post_y[:,0], post_y[:,1], post_y[:,2], color='g')
    ax4.plot(post_z2[:,0], post_z2[:,1], post_z2[:,2], color='b')
    ax4.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='k')

    ax5.plot(route[:,0], route[:,1], route[:,2])
    ax5.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
    #ax5.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='r')
    for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
        ax.set_aspect('equal')
else:
    print('check param plot type')



plt.show()
