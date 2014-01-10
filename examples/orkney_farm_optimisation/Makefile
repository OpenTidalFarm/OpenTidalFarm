.PHONY: mesh 

unsteady: clean
	unbuffer mpirun -n 4 python sw.py > unsteady_output.txt

unsteady_farm1: clean
	unbuffer mpirun -n 2 python sw.py 1 > unsteady_farm1_output.txt

unsteady_farm2: clean
	unbuffer mpirun -n 2 python sw.py 2 > unsteady_farm2_output.txt

unsteady_farm3: clean
	unbuffer mpirun -n 2 python sw.py 3 > unsteady_farm3_output.txt

unsteady_farm4: clean
	unbuffer mpirun -n 2 python sw.py 4 > unsteady_farm4_output.txt

multisteady: clean
	unbuffer mpirun -n 4 python sw_multisteady.py > multisteady_output.txt

multisteady_masked: clean
	unbuffer mpirun -n 4 python sw_multisteady_masked.py > multisteady_masked_output.txt

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

