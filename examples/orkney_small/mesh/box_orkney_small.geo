lat1 = 58.5;
lat2 = 59; 
lon1 = -3;
lon2 = -2.5;
pi = 3.14159265359;

Field[1] = Box;
Field[1].VOut = -1;
Field[1].XMax = lon2/360*2*pi;
Field[1].XMin = lon1/360*2*pi;
Field[1].YMax = lat2/360*2*pi; 
Field[1].YMin = lat1/360*2*pi;
Field[1].ZMax = 10000000000;
Field[1].ZMin = -10000000000;
Field[2] = LonLat;
