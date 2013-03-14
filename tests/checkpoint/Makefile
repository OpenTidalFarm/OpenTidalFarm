run: clean
	mpirun -n 2 python sw.py
	mpirun -n 2 python sw.py --from-checkpoint
	mpirun -n 2 python sw.py --from-checkpoint --high-tol
clean:
	rm -f *vtu
	rm -f *pvd
	rm -f *dat


