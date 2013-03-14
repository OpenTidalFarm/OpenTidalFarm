mesh:
	@echo "Generating coarse mesh..."
	gmsh -2 earth_orkney_converted_coarse.geo
	dolfin-convert -o xml earth_orkney_converted_coarse.msh earth_orkney_converted_coarse.xml
	../../../scripts/convert_to_new_xml.py earth_orkney_converted_coarse.xml
