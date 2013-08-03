iter_plot:
	python iter_plot.py output.txt 

generate_power_timeline:
	cat output.txt | grep Power | tail -n 50 | cut -c38- > power_timeline.txt
	python plot_power_timeline.py


run: clean 
	python sw.py > output.txt

mesh:	
	gmsh -2 mesh.geo
	dolfin-convert -o xml mesh.msh mesh.xml
	../../scripts/convert_to_new_xml.py

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

