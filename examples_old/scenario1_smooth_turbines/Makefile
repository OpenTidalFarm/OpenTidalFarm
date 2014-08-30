run: clean 
	unbuffer time mpirun -n 2 python sw.py > output.txt

coarse: clean 
	python sw_coarse.py 

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py
	gmsh -2 mesh_coarse.geo
	dolfin-convert -o xml mesh_coarse.msh mesh_coarse.xml
	../../scripts/convert_to_new_xml.py mesh_coarse.xml

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

