#!/usr/bin/python
from dolfin import *
mesh = Mesh('mesh.xml')
mesh_function = MeshFunction('uint', mesh, "mesh_facet_region.xml")
file = File('mesh_facet_region.xml')
file << mesh_function
