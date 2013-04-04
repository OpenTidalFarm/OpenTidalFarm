run: clean 
	unbuffer time mpirun -n 2 python sw.py mesh_coarse.xml > output_layer1.txt
	unbuffer time mpirun -n 2 python sw.py mesh.xml --from-checkpoint > output_layer1.txt

coarse: clean 
	python sw_coarse.py 

mesh:	
	for number in 0 1 2 ; do \
		gmsh -2 mesh$$number.geo; \
		dolfin-convert -o xml mesh$$number.msh mesh$$number.xml; \
		../../scripts/convert_to_new_xml.py mesh$$number.xml; \
	done

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

