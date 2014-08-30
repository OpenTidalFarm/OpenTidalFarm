run: clean 
	unbuffer time mpirun -n 2 python sw.py > output.txt

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

