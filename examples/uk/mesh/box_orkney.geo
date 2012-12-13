lon1 = 58.390773;
lon2 = 59.696025; 
lat1 = -4.290161;
lat2 = -1.192017;
pi = 3.14159265359;

Field[1] = Box;
Field[1].VOut = -1;
Field[1].XMax = lat2/360*2*pi;
Field[1].XMin = lat1/360*2*pi;
Field[1].YMax = lon2/360*2*pi; 
Field[1].YMin = lon1/360*2*pi;
Field[1].ZMax = 10000000000;
Field[1].ZMin = -10000000000;
Field[2] = LonLat;
