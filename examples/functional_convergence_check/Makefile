run: clean
	python sw.py 100. -- 33. 33.5 34. 34.5 35. 
	python sw.py 100.5 -- 33. 33.5 34. 34.5 35. 
	python sw.py 101. -- 33. 33.5 34. 34.5 35. 
	python sw.py 101.5 -- 33. 33.5 34. 34.5 35. 
	python sw.py 102. -- 33. 33.5 34. 34.5 35. 
	python sw.py --fine 100. -- 33. 33.5 34. 34.5 35. 
	python sw.py --fine 100.5 -- 33. 33.5 34. 34.5 35. 
	python sw.py --fine 101. -- 33. 33.5 34. 34.5 35. 
	python sw.py --fine 101.5 -- 33. 33.5 34. 34.5 35. 
	python sw.py --fine 102. -- 33. 33.5 34. 34.5 35. 
clean:
	rm -f functional_plot.png
	rm -f *vtu
	rm -f *pvd
