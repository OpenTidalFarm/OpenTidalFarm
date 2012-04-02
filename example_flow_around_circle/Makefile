default:
	gmsh -2 mesh.geo
	dolfin-convert mesh.msh mesh.xml
	mpirun -n 4 python sw.py
