lat1 = 58.390773;
lat2 = 59.696025; 
lon1 = -4.290161;
lon2 = -1.192017;
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
