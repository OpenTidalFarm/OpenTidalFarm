.PHONY: mesh 

run: clean 
	annotate-output mpirun -n 24 python sw.py > output.txt

restart: clean 
	annotate-output mpirun -n 24 python sw.py --from-checkpoint > output.txt

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

