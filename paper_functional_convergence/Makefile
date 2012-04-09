run: clean
	parallel python sw.py -- 33 33.5 34 34.5 35 
	parallel python sw.py --fine -- 33 33.5 34 34.5 35 
clean:
	rm -f functional_plot.png
	rm -f *vtu
	rm -f *pvd
