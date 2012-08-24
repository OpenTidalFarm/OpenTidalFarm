run: clean
	python sinusoidal_scenario_spatial.py 
	python sinusoidal_scenario_temporal.py 

mesh:	
	gmsh -2 mesh_dummy.geo
	dolfin-convert -o xml mesh_dummy.msh mesh_dummy.xml
	../../scripts/convert_to_new_xml.py mesh_dummy.xml

clean:
	rm -f functional_plot.png
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh


