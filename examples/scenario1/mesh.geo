basin_x = 500;
basin_y = 300;
site_x = 450;
site_y = 200;
n = 50;
element_size = 1;
element_size_coarse = 10;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};
Point(3) = {0, basin_y, 0, element_size_coarse};
Point(4) = {basin_x, basin_y, 0, element_size_coarse};

Point(5) = {(basin_x - site_x)/2, (basin_y - site_y)/2, 0, element_size};
Extrude{site_x, 0, 0} { Point{5}; Layers{20}; }
Extrude{0, site_y, 0} { Line{1}; Layers{20}; }
//Point(6) = {site_x + (basin_x-site_x)/2, (basin_y - site_y)/2, 0, element_size};
//Line(10) = {5, 6};
//Extrude{0, site_y, 0} { Line{10}; Layers{20}; }
Line(6) = {1, 2};
Line(7) = {2, 4};
Line(8) = {4, 3};
Line(9) = {3, 1};
Line Loop(10) = {9, 6, 7, 8};
Line Loop(11) = {3, 2, -4, -1};
Plane Surface(12) = {10, 11};
Physical Surface(13) = {12, 5};
Physical Line(14) = {7};
Physical Line(15) = {9};
Physical Line(16) = {8, 6};
