headland.xml: headland.geo
	gmsh -2 headland.geo
	dolfin-convert headland.msh headland.xml
	convert_to_new_xml.py headland.xml
