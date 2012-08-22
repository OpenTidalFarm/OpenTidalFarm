tests:
	@echo "=================== Running wave_dirichlet test ==================="
	cd mms_wave_dirichlet; make
	@echo "=================== Running strong wave_dirichlet test ==================="
	cd mms_wave_strong_dirichlet; make
	@echo "=================== Running wave_flather test ==================="
	cd mms_wave_flather; make
	@echo "=================== Running wave_flather advection test ==================="
	cd mms_wave_advection_flather; make
	@echo "=================== Running wave_flather friction test ==================="
	cd mms_wave_friction_flather; make
	@echo "=================== Running wave_flather diffusion test ==================="
	cd mms_wave_diffusion_flather; make
	@echo "=================== Running partial_derivative_turbine test ==================="
	cd partial_derivative_turbine; make
	@echo "=================== Running friction gradient test ==================="
	cd gradient_friction; make
	@echo "=================== Running position gradient test with strong boundary conditions ==================="
	cd gradient_pos_strong_dirichlet; make
	@echo "=================== Running gradient test for all configurations ==================="
	cd gradient_configurations; make
	@echo "=================== Running optimal friction mini model test ==================="
	cd optimal_friction_mini_model; make
	@echo "=================== Running optimal position mini model test ==================="
	cd optimal_position_mini_model; make
	@echo "=================== Running optimal friction for single turbine test ==================="
	cd optimal_friction_single_turbine; make
	@echo "=================== Running constraint derivatives test ==================="
	cd constraint_derivatives; make
	@echo "=================== All tests passed ===================" 
