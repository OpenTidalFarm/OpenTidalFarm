run: clean
	time mpirun -n 2 python sw.py 

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py

clean:
	rm -f *vtu
	rm -f *pvd
