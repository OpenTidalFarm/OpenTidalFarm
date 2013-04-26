run: clean 
	for number in 2 1 0 ; do \
		unbuffer time mpirun -n 2 python sw.py $$number.xml > output_$$number.txt; \
	done

coarse: clean 
	python sw_coarse.py 

mesh:	
	for number in 2 1 0 ; do \
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

clean_checkpoints:
	rm -f *.dat
