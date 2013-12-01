mesh:
	@echo "Generating mesh..."
	gmsh -2 coast_idBoundary_utm_no_islands.geo
	dolfin-convert -o xml coast_idBoundary_utm_no_islands.msh coast_idBoundary_utm_no_islands.xml
	../../../scripts/convert_to_new_xml.py coast_idBoundary_utm_no_islands.xml
	@echo "Generating mesh..."
	gmsh -2 coast_idBoundary_utm_no_islands_individual_farm_ids.geo
	dolfin-convert -o xml coast_idBoundary_utm_no_islands_individual_farm_ids.msh coast_idBoundary_utm_no_islands_individual_farm_ids.xml
	../../../scripts/convert_to_new_xml.py coast_idBoundary_utm_no_islands_individual_farm_ids.xml
