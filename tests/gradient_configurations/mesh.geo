basin_x = 1200;
basin_y = 1000;
land_x = 600;
land_y = 300;
land_site_delta = 100;
site_x = 150;
site_y = 100;
//element_size = 1;
element_size = 10;
element_size_coarse = 50;

Point(1) = {0, basin_x-land_x, 0, element_size_coarse};
Point(2) = {basin_x-land_x, 0, 0, element_size_coarse};
Point(3) = {basin_x-land_x, land_y, 0, element_size_coarse};
Point(4) = {basin_x, land_y, 0, element_size_coarse};
Point(5) = {basin_x, basin_y, 0, element_size_coarse};
Point(6) = {basin_x/3, basin_y, 0, element_size_coarse};

Point(7) = {basin_x-land_x, land_y+land_site_delta, 0, element_size};
Extrude{site_x, 0, 0} { Point{7}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{1}; Layers{site_y/element_size}; }

Line(6) = {1, 2};
Line(7) = {2, 3};
Line(8) = {3, 4};
Line(9) = {4, 5};
BSpline(10) = {1, 6, 5};
Line Loop(11) = {10, -9, -8, -7, -6};
Line Loop(12) = {1, 4, -2, -3};
Plane Surface(13) = {11, 12};
Physical Surface(14) = {13, 5};
Physical Line(1) = {6};
Physical Line(3) = {7, 8, 10};
Physical Line(2) = {9};
