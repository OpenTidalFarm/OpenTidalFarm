tests:
	@echo "=================== Running wave_dirichlet test ==================="
	cd test_mms_wave_dirichlet; make
	@echo "=================== Running wave_flather test ==================="
	cd test_mms_wave_flather; make
	@echo "=================== Running taylor remainder turbine function test ==================="
	cd test_taylor_remainder_turbine_function; make
	@echo "=================== Running taylor remainder initial condition test ==================="
	cd test_taylor_remainder_initial_condition; make
	@echo "=================== Running taylor remainder friction test ==================="
	cd test_taylor_remainder_friction; make
	@echo "=================== Running functional convergence test ==================="
	cd test_functional_convergence; make
	@echo "=================== Running optimal friction for one turbine test ==================="
	cd test_optimal_friction_for_one_turbine; make
	@echo "=================== All tests passed ===================" 
