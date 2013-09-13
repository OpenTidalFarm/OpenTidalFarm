.PHONY: mesh 

run: clean 
	python sw.py

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

