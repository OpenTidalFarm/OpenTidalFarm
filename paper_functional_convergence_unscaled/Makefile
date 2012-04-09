run: clean
	parallel python sw.py -- 500 505 510 515 520 525 530 535 540 545 550 555 560 
	parallel python sw.py --fine -- 503 506 508 511 513 515
clean:
	rm -f functional_plot.png
	rm -f *vtu
	rm -f *pvd
