basin_x = 1200;
land_x = 600;
land_y = 300;
land_site_delta = 100;
site_x = 150;
site_y = 100;
basin_y = land_y + basin_x - land_x;
element_size = 1;
element_size_coarse = 10;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x-land_x, 0, 0, element_size_coarse};
Point(3) = {basin_x-land_x, land_y, 0, element_size_coarse};
Point(4) = {basin_x, land_y, 0, element_size_coarse};
Point(5) = {basin_x, basin_y, 0, element_size_coarse};
Point(6) = {0, land_y, 0, element_size_coarse};
Point(7) = {basin_x-land_x, basin_y, 0, element_size_coarse};

Point(8) = {basin_x-land_x, land_y+land_site_delta, 0, element_size};
Extrude{site_x, 0, 0} { Point{8}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{1}; Layers{site_y/element_size}; }

Line(6) = {1, 2};
Line(7) = {2, 3};
Line(8) = {3, 4};
Line(9) = {4, 5};
Line(10) = {1, 6};
Line(11) = {5, 7};
Circle(12) = {6, 3, 7};
Line Loop(13) = {10, 12, -11, -9, -8, -7, -6};
Physical Line(1) = {6};
Physical Line(2) = {9};
Physical Line(3) = {12, 8, 7, 10, 11};
Line Loop(14) = {2, -4, -1, 3};
Plane Surface(15) = {13, 14};
Physical Surface(16) = {15, 5};
