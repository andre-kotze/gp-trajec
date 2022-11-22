import numpy as np
from data.test_data_3d import line, test_hull_pts
import pygalmesh
#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d

line[:,0] += 30
line[:,2] -= 5
line[:,1] += 10

radius = 1.0
displacement = 0.5
s0 = pygalmesh.Ball([displacement, 0, 0], radius)
s1 = pygalmesh.Ball([-displacement, 0, 0], radius)
i = pygalmesh.Difference()
print(len(i))




