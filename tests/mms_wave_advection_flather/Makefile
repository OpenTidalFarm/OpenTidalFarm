run: clean
	@echo "Running spatial convergence test Newton solver"
	mpirun -n 4 python sw_newton.py
	@echo "Running temporal convergence test"
	mpirun -n 4 python sw_time.py
clean:
	rm -f *vtu
	rm -f *pvd
