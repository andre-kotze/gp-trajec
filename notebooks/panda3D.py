import panda3D
from gptrajec import faces_from_poly
from shapely.geometry import Polygon

ground = CollisionPlane(Plane(Vec3(0, 0, 1), Point3(0, 0, 0)))

segment = CollisionSegment(ax, ay, az, bx, by, bz)

quad = CollisionPolygon(Point3(0, 0, 0), Point3(0, 0, 1),
                        Point3(0, 1, 1), Point3(0, 1, 0))

def polys_from_poly(pointslist):
    polygons = [polygon]
    coords = polygon.exterior.coords
    for pt in range(len(coords) - 1):
        poly = [coords[pt], coords[pt + 1]]
        polybase = [(tpl[0],tpl[1],0) for tpl in poly]
        poly.extend(polybase.reverse())
        polygons.append(Polygon(poly))

    return polygons

building = Polygon([
    (0,10,15),
    (5,10,15),
    (5,5,15),
    (0,5,15)])

military = Polygon([
    (15,10,10),
    (25,10,15),
    (25,0,10)])

nofly = Polygon([
    (0,25,30),
    (15,25,30),
    (15,20,30),
    (0,20,30)])

barriers = []
for bar in [building,military,nofly]:
    barriers.extend(faces_from_poly(bar))
# now barriers is a list of shapely polygons


def inters_check():
    

inters_check()