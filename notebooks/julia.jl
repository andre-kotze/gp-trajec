#using Meshes

a = Point(1.1, 1.1, 1.1)
b = Point(1.9, 1.1, 1.1)
c = Point(1.1, 1.9, 1.1)
d = Point(1.1, 1.1, 1.9)
mytetraedron = Primitive(a, b, c, d)
typeof(mytetrahedron)

println("Hello World")