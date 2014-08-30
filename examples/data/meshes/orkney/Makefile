mesh:
	gmsh -2 orkney_utm.geo
	dolfin-convert -o xml orkney_utm.msh orkney_utm.xml
	../../../scripts/convert_to_new_xml.py orkney_utm.xml
