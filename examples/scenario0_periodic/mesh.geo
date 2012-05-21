site_x = 100;
site_y = 50;
element_size = 1.0;

Point(5) = {0, 0, 0, element_size};
Extrude{site_x, 0, 0} { Point{5}; Layers{site_x/element_size}; }
Extrude{0, site_y, 0} { Line{1}; Layers{site_y/element_size}; }

Line Loop(10) = {9, 6, 7, 8};
Physical Line(2) = {4};
Physical Line(1) = {3};
Physical Line(3) = {1, 2};
Physical Surface(11) = {5};
