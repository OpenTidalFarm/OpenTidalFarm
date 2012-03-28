run: clean
	@echo "Running spatial convergence test"
	mpirun -n 6 python sw.py
	@echo "Running temporal convergence test"
	mpirun -n 6 python sw_time.py
clean:
	rm -f *vtu
	rm -f *pvd


