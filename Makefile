tests:
	@echo "=================== Running wave_dirichlet test ==================="
	cd test_mms_wave_dirichlet; make
	@echo "=================== Running wave_flather test ==================="
	cd test_mms_wave_flather; make
	@echo "=================== Running wave divett_adjoint test ==================="
	cd test_taylor_remainder_initial_condition; make
	@echo "=================== Running taylor remainder test ==================="
	cd test_taylor_remainder; make
	@echo "=================== Running functional convergence test ==================="
	cd test_functional_convergence; make
	@echo "=================== All tests passed ===================" 
