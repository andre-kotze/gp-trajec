import numpy as np
import pyvista as pv

rng = np.random.default_rng()

# create dummy data
N = 10
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
radii = rng.uniform(0.5, 1.5, N)
coords = np.array([np.cos(angles), np.sin(angles)]) * radii
points_2d = coords.T  # shape (N, 2)

# embed in 3d, create filled polygon
points_3d = np.pad(points_2d, [(0, 0), (0, 1)])  # shape (N, 3)
face = [N + 1] + list(range(N)) + [0]  # cell connectivity for a single cell
polygon = pv.PolyData(points_3d, faces=face)

# extrude along z and plot
body = polygon.extrude((0, 0, 0.5), capping=True)
body.plot(color='white', specular=1, screenshot='extruded.png')