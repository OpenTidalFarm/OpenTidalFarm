
run: clean mesh
	mpirun -n 2 python sw.py
clean:
	rm -f *vtu
	rm -f *pvd

mesh:
	gmsh -2 mesh.geo
	dolfin-convert mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py	

