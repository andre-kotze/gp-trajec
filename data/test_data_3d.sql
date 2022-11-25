SELECT ST_3DIntersection(linestring, polygon) As wkt
FROM  ST_GeomFromText('LINESTRING Z (2 2 6,1.5 1.5 7,1 1 8,0.5 0.5 8,0 0 10)') AS linestring
 CROSS JOIN ST_GeomFromText('POLYGON((0 0 8, 0 1 8, 1 1 8, 1 0 8, 0 0 8))') AS polygon;

/** polyhedron **/
ST_GeomFromText('POLYGON((0 0 8, 0 1 8, 1 1 8, 1 0 8, 0 0 8))')

/** non-intersecting lineZ **/
ST_GeomFromText('LINESTRING Z (2 2 6,1.5 1.5 7,1 1 7.5,0.5 0.5 7.5,0 0 6)')

/** line-intersecting lineZ **/
ST_GeomFromText('LINESTRING Z (2 2 6,1.5 1.5 7,1 1 8,0.5 0.5 8,0 0 10)')

/** point-intersecting lineZ **/
ST_GeomFromText('LINESTRING Z (2 2 6,1.5 1.5 7,1 1 8,0.5 0.5 9,0 0 10)')