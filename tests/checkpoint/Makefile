run: clean
	mpirun -n 2 python sw.py
	mpirun -n 2 python sw.py --from-checkpoint
clean:
	rm -f *vtu
	rm -f *pvd
	rm -f *dat


