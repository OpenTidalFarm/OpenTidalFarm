tests:
	@echo "=================== Running wave_dirichlet test ==================="
	cd wave_dirichlet; make
	@echo "=================== Running wave_flather test ==================="
	cd wave_flather; make
	@echo "=================== Running wave divett_adjoint test ==================="
	cd divett_adjoint; make
