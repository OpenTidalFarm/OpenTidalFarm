from math import asin, atan2, sqrt, pi
import sys
import re
from IPython import embed

def RadtoDeg(x):
     return x*180./pi

def Cartesian2LatLon(x, y, z):
   R = sqrt(x**2 + y**2 + z**2)
   lat = RadtoDeg(asin(z / R))
   lon = RadtoDeg(atan2(y, x))
   return (lat, lon)

def Stereographic2Cartesian(x, y):
    return (2.*x/(1+x**2+y**2), 2.*y/(1+x**2+y**2), (-1.+x**2+y**2)/(1+x**2+y**2))

def Stereographic2LatLon(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    return Cartesian2LatLon(x, y, z)

def main():
    if len(sys.argv) != 2:
        print "Converts the points in a .geo file from stereographic coordinates a flat coordaintes system."
        print "Usage: convert.py file.geo"
        sys.exit(1)

    infile = sys.argv[1]
    outfile = "_converted.".join(infile.split("."))
    fin = open(infile, "r")
    fout = open(outfile, "w")
    maxlat, minlat = -1000, 1000
    maxlon, minlon = -1000, 1000

    for line in fin:
        if "Point" in line:
            try:
                m = re.search('{([-0-9\.e]*), ([-0-9\.e]*), ([-0-9\.e]*) }', line)
                assert(float(m.group(3)) == 0)
                x, y = (float(m.group(1)), float(m.group(2)))
            except:
                print "Error while parsing line: ", line
                embed()
                sys.exit(1)
            lat, lon = Stereographic2LatLon(x, y)
            # Need some conversions
            lat = - lat # 
            lon = 180 -lon
            maxlat = max(maxlat, lat)
            maxlon = max(maxlon, lon)
            minlat = min(minlat, lat)
            minlon = min(minlon, lon)
            print lat, lon
        fout.write(line)
    print "maxlat: ", maxlat
    print "minlat: ", minlat
    print "maxlon: ", maxlon
    print "minlon: ", minlon
    

if __name__ == "__main__":
    main()
