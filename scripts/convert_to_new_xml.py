#!/usr/bin/python
from dolfin import *
import sys
if len(sys.argv) > 1:
	meshfile = sys.argv[1]
else:
	meshfile = "mesh.xml"
mesh = Mesh(meshfile)
mesh_function = MeshFunction('size_t', mesh, meshfile[:-4] + "_facet_region.xml")
file = File(meshfile[:-4] + '_facet_region.xml')
file << mesh_function

try:
	mesh_function = MeshFunction('size_t', mesh, meshfile[:-4] + "_physical_region.xml")
	file = File(meshfile[:-4] + '_physical_region.xml')
	file << mesh_function
except IOError:
	print "Could not find a physical region file. Skipping."
