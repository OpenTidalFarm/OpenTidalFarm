.PHONY: mesh 

unsteady: clean
	unbuffer mpirun -n 8 python sw.py > unsteady_output.txt

multisteady: clean
	unbuffer mpirun -n 8 python sw_multisteady.py > multisteady_output.txt

multisteady_masked: clean
	unbuffer mpirun -n 8 python sw_multisteady_masked.py > multisteady_masked_output.txt

mesh:	
	cd mesh; make mesh

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html
	rm -Rf iter_*

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

