basin_x = 1600;
land_x = 640;
land_y = 320;
land_site_delta = 100;
site_x = 320;
site_y = 160;
basin_y = land_y + basin_x - land_x;
small_circle_diameter = 100;
element_size = 2;
element_size_coarse = 20;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x-land_x, 0, 0, element_size_coarse};
Point(3) = {basin_x-land_x, land_y, 0, element_size_coarse};
Point(4) = {basin_x, land_y, 0, element_size_coarse};
Point(5) = {basin_x, basin_y, 0, element_size_coarse};
Point(6) = {basin_x-land_x, basin_y, 0, element_size_coarse};
Point(7) = {0, land_y, 0, element_size_coarse};
Point(8) = {basin_x-land_x, land_y-small_circle_diameter, 0, element_size_coarse};
Point(9) = {basin_x-land_x+small_circle_diameter, land_y, 0, element_size_coarse};
Point(10) = {basin_x-land_x+small_circle_diameter, land_y-small_circle_diameter, 0, element_size_coarse};

Point(11) = {basin_x-land_x, land_y+land_site_delta, 0, element_size};
Extrude{site_x, 0, 0} { Point{11}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{1}; Layers{site_y/element_size}; }
Line(6) = {1, 2};
Line(7) = {2, 8};
Circle(8) = {8, 10, 9};
Line(9) = {9, 4};
Line(10) = {4, 5};
Line(11) = {5, 6};
Circle(12) = {6, 3, 7};
Line(13) = {7, 1};
Line Loop(14) = {12, 13, 6, 7, 8, 9, 10, 11};
Line Loop(15) = {2, -4, -1, 3};
Plane Surface(16) = {14, 15};
Physical Line(1) = {6};
Physical Line(2) = {10};
Physical Line(3) = {9, 8, 7, 13, 12, 11};
Physical Surface(20) = {16, 5};
