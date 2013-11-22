coarse: clean 
	python sw_coarse.py 

mesh:	
	gmsh -2 mesh_coarse.geo
	dolfin-convert -o xml mesh_coarse.msh mesh_coarse.xml
	../../scripts/convert_to_new_xml.py mesh_coarse.xml

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html
	rm -Rf iter_*
	rm -Rf xyz_data

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

