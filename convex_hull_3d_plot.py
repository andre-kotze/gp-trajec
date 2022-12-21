import numpy as np
import cdd as pcdd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas
import geopandas as gpd
import datetime as dt
import time
import fiona
from scipy.spatial import ConvexHull

#from shapely import from_wkt
from shapely.geometry import MultiPolygon, MultiLineString, LineString, shape, Point
from data.test_data_2d import barriers, pts, example_route, example_space, islas
START, END = pts['bc'], pts['tc']

example_space = []
with fiona.open('./data/example_space_z_added.gpkg', layer='z_added') as layer:
    for n, feat in enumerate(layer):
        example_space.append(shape(feat['geometry']).geoms[0])
        #example_space.append(feat)
        if n == 0:
            print(type(feat))
example_space = MultiPolygon(example_space)
#print(example_space.geoms[0])

testline = LineString(((324400, 59800, 400),(326000, 60500, 200)))
def convex_hull(poly):
    high_pts = np.array(poly.exterior.coords)
    x, y = poly.exterior.xy
    z = np.zeros(len(x))
    print(f'{len(x)=}')
    low_pts = np.column_stack((x,y,z))
    pts = np.append(high_pts, low_pts, axis=0)
    return ConvexHull(pts)

#example_space.geoms[0].exterior.x
hull = convex_hull(example_space.geoms[3])






points= hull.points
num_verts = len(points)
print(f'{num_verts=}')

# to get the convex hull with cdd, one has to prepend a column of ones
vertices = np.hstack((np.ones((num_verts,1)), points))

# do the polyhedron
mat = pcdd.Matrix(vertices, linear=False, number_type="fraction") 
mat.rep_type = pcdd.RepType.GENERATOR
poly = pcdd.Polyhedron(mat)

# get the adjacent vertices of each vertex
adjacencies = [list(x) for x in poly.get_input_adjacency()]

# store the edges in a matrix (giving the indices of the points)
edges = [None]*(num_verts-1)
for i,indices in enumerate(adjacencies[:-1]):
    indices = list(filter(lambda x: x>i, indices))
    l = len(indices)
    col1 = np.full((l, 1), i)
    indices = np.reshape(indices, (l, 1))
    edges[i] = np.hstack((col1, indices))
Edges = np.vstack(tuple(edges)).astype(int)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

print(type(Edges[:,0]))
print(type(Edges[:,0][0]))

start = points[Edges[:,0]]
end = points[Edges[:,1]]

for i in range(len(Edges)):
    ax.plot(
        [start[i,0], end[i,0]], 
        [start[i,1], end[i,1]], 
        [start[i,2], end[i,2]],
        "blue"
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

#ax.set_xlim3d(-1,5)
#ax.set_ylim3d(-1,5)
#ax.set_zlim3d(-1,5)
#
plt.show()