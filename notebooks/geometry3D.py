from Geometry3D import *

# a wrapper to generate boxes, given 6 extent values
def cuboid(ext):
    xmin, xmax, ymin, ymax, zmin, zmax = ext
    a = Point(xmax,ymax,zmax)
    b = Point(xmin,ymax,zmax)
    c = Point(xmin,ymin,zmax)
    d = Point(xmax,ymin,zmax)
    e = Point(xmax,ymax,zmin)
    f = Point(xmin,ymax,zmin)
    g = Point(xmin,ymin,zmin)
    h = Point(xmax,ymin,zmin)
    cpg0 = ConvexPolygon((a,d,h,e))
    cpg1 = ConvexPolygon((a,e,f,b))
    cpg2 = ConvexPolygon((c,b,f,g))
    cpg3 = ConvexPolygon((c,g,h,d))
    cpg4 = ConvexPolygon((a,b,c,d))
    cpg5 = ConvexPolygon((e,h,g,f))
    return ConvexPolyhedron((cpg0,cpg1,cpg2,cpg3,cpg4,cpg5))

def prism(corners):
    pass
    # here have to draw some non-parallelepiped geoms

'''
building = ConvexPolygon((
    Point(0,10,0),
    Point(5,10,0),
    Point(5,5,0),
    Point(0,5,0),
    Point(0,10,15),
    Point(5,10,15),
    Point(5,5,15),
    Point(0,5,15)
        ))

military = ConvexPolygon((
    Point(15,10,0),
    Point(25,10,0),
    Point(25,0,0),
    Point(15,10,10),
    Point(25,10,15),
    Point(25,0,10)
    ))

nofly = ConvexPolygon((
    Point(0,25,0),
    Point(15,25,0),
    Point(15,20,0),
    Point(0,20,0),
    Point(0,25,30),
    Point(15,25,30),
    Point(15,20,30),
    Point(0,20,30)
    ))
'''


def inters_ex():
    building = cuboid((0,5,5,10,0,15))
    military = cuboid((15,25,0,10,0,15))
    nofly = cuboid((0,15,20,25,0,30))
    r = Renderer()
    r.add((building,'r',1),normal_length = 0)
    r.add((military,'r',1),normal_length = 0)
    r.add((nofly,'r',1),normal_length = 0)
    r.show()

inters_ex()