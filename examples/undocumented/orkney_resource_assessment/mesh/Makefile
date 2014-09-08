mesh:
	@echo "Generating mesh..."
	gmsh -2 orkney.geo
	dolfin-convert -o xml orkney.msh orkney.xml
	../../../scripts/convert_to_new_xml.py orkney.xml
