import json
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

y = np.linspace(0,2*np.pi**1.4,100)
x = np.sin(1)*y + np.sin(np.cos(y)) * np.sin(y) -np.cos(1) - np.cos(np.cos(-1-y))
z = np.sin(np.cos(np.sin(x))) * np.sin(np.cos(x)) + np.sin(1)

fig = plt.figure()


geojson = '{"type":"Polygon","coordinates":[[[0,0,4],[0,1,6],[1,1,6],[1,0,8],[0,0,4]]]}'
geojson = geojson.replace('""', '"')
geo_dict = json.loads(geojson)
x, y, z = [], [], []
for xc, yc, zc in geo_dict['coordinates'][0]:
    x.append(xc)
    y.append(yc)
    z.append(zc)

ax0 = fig.add_subplot(111, projection='3d')
ax0.set_title('3D Geometry Viewer')
#ax0.set_aspect('equal')
ax0.plot(x,y,z, color='r')
    

plt.show()
