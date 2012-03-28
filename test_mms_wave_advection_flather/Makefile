run: clean
	@echo "Running spatial convergence test Newton solver"
	mpirun -n 6 python sw_newton.py
	@echo "Running spatial convergence test with Picard solver"
	mpirun -n 6 python sw_picard.py
	@echo "Running temporal convergence test"
	mpirun -n 6 python sw_time.py
clean:
	rm -f *vtu
	rm -f *pvd


