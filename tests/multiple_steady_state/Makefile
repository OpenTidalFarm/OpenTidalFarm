run: clean mesh
	time mpirun -n 2 python multi_steady_state_2steps.py
	time mpirun -n 2 python multi_steady_state_1step.py 

mesh:	
	gmsh -2 mesh_coarse.geo
	dolfin-convert -o xml mesh_coarse.msh mesh_coarse.xml
	../../scripts/convert_to_new_xml.py mesh_coarse.xml

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html
	rm -fR iter_*

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh
