.PHONY: mesh 

run: clean 
	unbuffer time mpirun -n 2 python sw.py > output.txt

mesh:	
	cd mesh; make mesh

clean:
	rm -f *vtu
	rm -f *pvd
	rm -f output.txt
	rm -f *.html

clean_mesh:
	rm -f *.xml
	rm -f *.xml.bak
	rm -f *.msh

