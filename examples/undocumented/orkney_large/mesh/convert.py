from math import asin, atan2, sqrt, pi, sin
import sys
import re
from IPython import embed
reference_lat = None

def Rad2Deg(x):
     return x*180./pi

def Deg2Rad(x):
     return x/180.*pi

def LatLon2Flat(lat, lon):
   global reference_lat
   if reference_lat == None:
      reference_lat = lat

   earth_radius = 6371000.
   equator = 2 * pi * earth_radius
   r = equator * sin(Deg2Rad(90. - reference_lat))
   x = lon/360 * r 
   y = - lat/360 * equator 
   return x, y

def Cartesian2LatLon(x, y, z):
   R = sqrt(x**2 + y**2 + z**2)
   lat = Rad2Deg(asin(z / R))
   lon = Rad2Deg(atan2(y, x))
   return lat, lon

def Stereographic2Cartesian(x, y):
    return (2.*x/(1+x**2+y**2), 2.*y/(1+x**2+y**2), (-1.+x**2+y**2)/(1+x**2+y**2))

def Stereographic2LatLon(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    lat, lon = Cartesian2LatLon(x, y, z)
    # Need to do some conversions
    lat = - lat 
    lon = 180 - lon
    return lat, lon

def Stereographic2Flat(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    lat, lon = Cartesian2LatLon(x, y, z)
    return LatLon2Flat(lat, lon)

def main():
    if len(sys.argv) != 2:
        print "Converts the points in a .geo file from stereographic coordinates a flat coordaintes system."
        print "Usage: convert.py file.geo"
        sys.exit(1)

    infile = sys.argv[1]
    outfile = "_converted.".join(infile.split("."))
    fin = open(infile, "r")
    fout = open(outfile, "w")
    maxx, minx = -1e20, 1e20
    maxy, miny = -1e20, 1e20

    for line in fin:
        if "Point" in line:
            try:
                m = re.search('{([-0-9\.e]*), ([-0-9\.e]*), ([-0-9\.e]*) }', line)
                assert(float(m.group(3)) == 0)
                x, y = (float(m.group(1)), float(m.group(2)))
            except:
                print "Error while parsing line: ", line
                sys.exit(1)
            x, y = Stereographic2Flat(x, y)
            line = line.replace(m.group(1), str(x))
            line = line.replace(m.group(2), str(y))
            maxx = max(maxx, x)
            maxy = max(maxy, y)
            minx = min(minx, x)
            miny = min(miny, y)
        fout.write(line)
    print "New data range: [%f, %f] - [%f, %f]." % (minx, maxx, miny, maxy)
    print "Finished."

if __name__ == "__main__":
    main()
