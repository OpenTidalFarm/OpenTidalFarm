mesh:
	@echo "Generating mesh..."
	gmsh -2 earth_orkney_converted.geo
	dolfin-convert -o xml earth_orkney_converted.msh earth_orkney_converted.xml
	../../../scripts/convert_to_new_xml.py earth_orkney_converted.xml
