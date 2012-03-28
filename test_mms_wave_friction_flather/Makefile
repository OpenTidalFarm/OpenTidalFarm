run: clean
	@echo "Running convergence test with linear friction"
	mpirun -n 2 python sw_linear.py
	@echo "Running convergence test with quadratic friction"
	mpirun -n 2 python sw_quadratic.py
clean:
	rm -f *vtu
	rm -f *pvd


