test:
	@make clean
	python tests.py
	@make clean

clean:
	rm -rf *pvd *vtu

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py mesh.xml

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh
