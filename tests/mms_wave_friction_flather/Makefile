run: clean
	@echo "Running convergence test with linear friction"
	mpirun -n 2 python sw_linear.py
	@echo "Running convergence test with quadratic friction with Newton solver"
	mpirun -n 2 python sw_quadratic_newton.py
	@echo "Running convergence test with quadratic friction with Picard solver"
	mpirun -n 2 python sw_quadratic_picard.py
clean:
	rm -f *vtu
	rm -f *pvd


