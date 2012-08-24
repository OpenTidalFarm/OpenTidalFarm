basin_x = 3000;
basin_y = 1000;
n = 4;
element_size_coarse = basin_y/n;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};
Point(3) = {0, basin_y, 0, element_size_coarse};
Point(4) = {basin_x, basin_y, 0, element_size_coarse};

Line(6) = {1, 2};
Line(7) = {2, 4};
Line(8) = {4, 3};
Line(9) = {3, 1};
Line Loop(10) = {9, 6, 7, 8};
Plane Surface(12) = {10};
Physical Surface(13) = {12, 5};
Physical Line(2) = {7};
Physical Line(1) = {9};
Physical Line(3) = {8, 6};
