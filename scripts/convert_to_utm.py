#!/usr/bin/python
''' Converts a .geo file from lat/lon coordinates to UTM coordinates. '''

from math import asin, atan2, sqrt, pi, sin
import sys
import re
import utm

def Rad2Deg(x):
     return x*180./pi

def Deg2Rad(x):
     return x/180.*pi

def Cartesian2LatLon(x, y, z):
    R = sqrt(x**2 + y**2 + z**2)
    lat = Rad2Deg(asin(z / R))
    lon = Rad2Deg(atan2(y, x))

    # Peform some necessary conversions
    lat = - lat 
    lon = -(180 - lon)
    return lat, lon

def Stereographic2Cartesian(x, y):
    return (2.*x/(1+x**2+y**2), 2.*y/(1+x**2+y**2), (-1.+x**2+y**2)/(1+x**2+y**2))

def Stereographic2LatLon(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    lat, lon = Cartesian2LatLon(x, y, z)
    return lat, lon

def Stereographic2Flat(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    lat, lon = Cartesian2LatLon(x, y, z)
    return LatLon2Flat(lat, lon)

def LatLon2UTM(lat, lon):
    return utm.from_latlon(lat, lon)

def Stereographic2UTM(x, y):
    x, y, z = Stereographic2Cartesian(x, y)
    lat, lon = Cartesian2LatLon(x, y, z)
    return utm.from_latlon(lat, lon)

def main():
    if len(sys.argv) != 2:
        print "Converts the points in a .geo file from stereographic coordinates a flat coordinate system."
        print "Usage: convert.py file.geo"
        sys.exit(1)

    infile = sys.argv[1]
    outfile = "_utm.".join(infile.split("."))
    fin = open(infile, "r")
    fout = open(outfile, "w")

    zone_numbers = []
    zone_letters = []

    for line in fin:
        if "Point" in line:
            try:
                m = re.search('{\s*([-0-9\.e]*)\s*,\s*([-0-9\.e]*)\s*,\s*([-0-9\.e]*)\s*}', line)
                assert(float(m.group(3)) == 0)
                x, y = (float(m.group(1)), float(m.group(2)))
            except:
                print "Error while parsing line: ", line
                sys.exit(1)
            east, north, zone_number, zone_letter = LatLon2UTM(y, x)
            zone_letters.append(zone_letter)
            zone_numbers.append(zone_number)

            line = line.replace(m.group(1), str(east))
            line = line.replace(m.group(2), str(north))

        fout.write(line)
    if not len(set(zone_numbers)) == 1:
          print "Coordinates are in more than one UTM zone number, which is not yet supported."
          sys.exit(1)
    else:
          print "UTM zone number: ", zone_numbers[0]
    if not len(set(zone_letters)) == 1:
          print "Coordinates are in more than one UTM zone letter, which is not yet supported."
          sys.exit(1)
    else:
          print "UTM zone letter: ", zone_letters[0]

    print "Finished."

if __name__ == "__main__":
    main()
