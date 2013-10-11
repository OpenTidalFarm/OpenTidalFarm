mesh:
	@echo "Generating mesh..."
	gmsh -2 coast_idBoundary_utm.geo 
	dolfin-convert -o xml coast_idBoundary_utm.msh coast_idBoundary_utm.xml
	../../../scripts/convert_to_new_xml.py coast_idBoundary_utm.xml
