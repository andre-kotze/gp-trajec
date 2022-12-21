from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from gptrajec import transform_3d
#from trans3d import transform_3d as transform_zyz

plot_types = ['steps', 'party', 'result', 'flat']
plot_type = plot_types[0]

y = np.linspace(0,1,100)
#x = np.linspace(0,1,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
#y = 2*np.sin(x) + np.sin(4*x) + 3*np.cos(x) + np.sin(2*x) + np.sin(x**2) - np.cos(6*x)
z = np.sin(np.cos(np.sin(y))) * np.sin(np.cos(y)) + np.sin(1)
#z = -np.sin(1)*x - np.sin(np.cos(x)) * np.sin(x) -np.cos(1) - np.cos(np.cos(-1-x))

fig = plt.figure()

dest_int = np.array([[288_000,48_600,100],[292_600,48_960,0]])
dest_int_tf = dest_int - np.array([[288_000,48_600,10],[288_000,48_600,10]])
dest_int_norm = dest_int_tf / norm(dest_int_tf)
line = np.array([[xc,yc,zc] for xc,yc,zc in zip(x,y,z)])
#print(f'transforming route with {line.shape=}\nfrom {np.round(line[0],2)} to {np.round(line[-1],2)}')
route, steps, rotvec = transform_3d(line, dest_int, intermediates=True)
#route, steps = transform_zyz(line, dest_int, printing=True)
transed, normed, roted, post_scale = steps
(px, py, pz), (qx, qy, qz) = normed
p, q = normed
#post_scale, post_z, post_y, post_x = None, None, None, None
#post_scale = steps
#print(f'transformed route with {route.shape=}\nfrom {np.round(route[0],2)} to {np.round(route[-1],2)}')
#print(f'offset from true endpoint: {norm(dest_int[-1] - route[-1])}\n({dest_int[-1]-route[-1]})')
#ax1.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
print(f'\n\nNormalised:\n\t{norm(dest_int_norm)=}\n\t{norm(p)=}\n\t{norm(q)=}')

print(f'{rotvec=}')
rotvec_norm = rotvec / norm(rotvec)

if plot_type == plot_types[0]:
    ax0 = fig.add_subplot(231, projection='3d')
    ax0.set_title('Solution curve')
    ax1 = fig.add_subplot(232, projection='3d')
    ax1.set_title('Translated')
    ax2 = fig.add_subplot(233, projection='3d')
    ax2.set_title('Normalised')
    ax3 = fig.add_subplot(234, projection='3d')
    ax3.set_title('Rotated')
    ax4 = fig.add_subplot(235, projection='3d')
    ax4.set_title('Scaled')
    ax5 = fig.add_subplot(236, projection='3d')
    ax5.set_title('Final route')

    ax0.plot(x,y,z)
    ax1.plot(transed[:,0], transed[:,1], transed[:,2])
    
    ax2.plot([0,px],[0,py],[0,pz], color='r')
    ax2.plot([0,qx],[0,qy],[0,qz], color='g')
    ax2.plot(dest_int_norm[:,0],dest_int_norm[:,1],dest_int_norm[:,2], color='m')
    ax2.scatter(rotvec_norm[0],rotvec_norm[1],rotvec_norm[2])

    ax3.plot(roted[:,0], roted[:,1], roted[:,2])
    ax3.plot(dest_int_norm[:,0],dest_int_norm[:,1],dest_int_norm[:,2], color='m')
    ax4.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2])
    ax5.plot(route[:,0], route[:,1], route[:,2])
    ax5.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
    for ax in [ax0,ax1,ax2,ax3,ax4]:
        ax.scatter(0,0,0, color='orange')    
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
    ax2.plot(post_x[:,0], post_x[:,1], post_x[:,2], color='b')
    ax3.plot(route[:,0], route[:,1], route[:,2])
    ax3.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='m')
    for ax in [ax2,ax3]:
        ax.set_aspect('equal')
elif plot_type == plot_types[2]:
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')
    ax0.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2], color='y')
    ax0.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='r')
    ax1.plot(route[:,0], route[:,1], route[:,2], color='g')
    ax1.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
    for ax in [ax0,ax1]:
        ax.set_aspect('equal')
elif plot_type == plot_types[3]: # flat
    ax0 = fig.add_subplot(231, projection='3d')
    ax0.set_title('Scaled solution curve')
    ax1 = fig.add_subplot(232)
    ax1.set_title('Rotation in z')
    ax2 = fig.add_subplot(233)
    ax2.set_title('Rotation in y')
    ax3 = fig.add_subplot(234)
    ax3.set_title('Rotation in x')
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

    ax3.plot(post_y[:,1], post_y[:,2], color='g')# y,z
    ax3.plot(post_x[:,1], post_x[:,2], color='b')# y,z
    ax3.scatter(dest_int_tf[:,1], dest_int_tf[:,2], color='b')
    ax3.set_xlabel('y')
    ax3.set_ylabel('z')

    ax4.plot(post_scale[:,0], post_scale[:,1], post_scale[:,2], color='y')
    ax4.plot(post_z[:,0], post_z[:,1], post_z[:,2], color='r')
    ax4.plot(post_y[:,0], post_y[:,1], post_y[:,2], color='g')
    ax4.plot(post_x[:,0], post_x[:,1], post_x[:,2], color='b')
    ax4.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='k')

    ax5.plot(route[:,0], route[:,1], route[:,2])
    ax5.scatter(dest_int[:,0],dest_int[:,1],dest_int[:,2], color='r')
    #ax5.scatter(dest_int_tf[:,0],dest_int_tf[:,1],dest_int_tf[:,2], color='r')
    for ax in [ax0,ax1,ax2,ax3,ax4,ax5]:
        ax.set_aspect('equal')
else:
    print('check param plot type')



plt.show()
