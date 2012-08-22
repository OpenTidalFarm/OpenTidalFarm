run: clean mesh
	time mpirun -n 4 python sw.py 

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py

clean:
	rm -f functional_plot.png
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh


