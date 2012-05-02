run: clean
	gmsh -2 mesh.geo
	dolfin-convert mesh.msh mesh.xml
	time mpirun -n 4 python sw.py 
clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt


