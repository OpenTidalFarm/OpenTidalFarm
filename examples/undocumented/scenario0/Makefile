run: clean
	unbuffer time mpirun -n 4 python sw.py 

functional_vs_friction_plot:
	unbuffer time mpirun -n 4 python functional_vs_friction_plot.py

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py

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


