#!/usr/bin/env python
from dolfin import *
from opentidalfarm import *

import sys

bathnc = sys.argv[1]
utm_zone = int(sys.argv[2])
utm_band = sys.argv[3]

meshfile = sys.argv[4]
output = sys.argv[5]

bathexpr = BathymetryDepthExpression(bathnc, utm_zone, utm_band)
mesh = Mesh(meshfile)

V = FunctionSpace(mesh, "CG", 1)
bath = interpolate(bathexpr, V)

File(output) << bath
