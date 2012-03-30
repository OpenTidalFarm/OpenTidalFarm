run: clean
	time mpirun -n 4 python sw.py
	@echo "Check that the time is less then 15s."
clean:
	rm -f *vtu
	rm -f *pvd


