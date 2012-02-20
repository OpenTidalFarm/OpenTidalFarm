tests:
	@echo "=================== Running wave_dirichlet test ==================="
	cd test_mms_wave_dirichlet; make
	@echo "=================== Running wave_flather test ==================="
	cd test_mms_wave_flather; make
	@echo "=================== Running turbine function derivative test ==================="
	cd test_derivative_turbine_function; make
	@echo "=================== Running initial condition derivative test ==================="
	cd test_derivative_initial_condition; make
	@echo "=================== Running friction derivative test ==================="
	cd test_derivative_friction; make
	@echo "=================== Running functional convergence test ==================="
	cd test_functional_convergence; make
	@echo "=================== Running optimal friction test ==================="
	cd test_optimal_friction; make
	@echo "=================== Running optimal friction for one turbine test ==================="
	cd test_optimal_friction_for_one_turbine; make
	@echo "=================== All tests passed ===================" 
