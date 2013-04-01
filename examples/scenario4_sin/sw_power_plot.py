from opentidalfarm import *
set_log_level(ERROR)

# Some domain information extracted from the geo file
basin_x = 1280.
basin_y = 640.+320.
site_x = 320.
site_y = 160.
rad = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y/2-rad)/2-site_y/2 
eta0 = (2.0+1e-10)/sqrt(9.81/50) # This will give a inflow velocity of 2m/s
config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1, 0], eta0=eta0)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['initial_condition'] = ConstantFlowInitialCondition(config) 
config.params['automatic_scaling'] = False

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
turbine_pos_array = [ 490.        ,   90.        ,  524.43987312,  117.46272757,
543.38328616,  140.72529627,  566.94642491,  159.29350815,
536.514582  ,   90.        ,  560.97445248,  116.42404389,
584.53759123,  134.99225576,  534.06221939,  230.        ,
575.17902587,   90.        ,  602.78441784,  158.80513712,
585.19325152,  183.10638951,  588.60725283,  230.        ,
617.94138862,   90.        ,  623.07910822,  180.8987005 ,
605.4879419 ,  205.19995289,  622.36863096,  230.        ,
668.06450844,   90.        ,  653.02805623,  182.64813272,
671.11778057,  206.58057667,  652.36863096,  230.        ,
707.38630967,   90.        ,  673.98296502,  161.17974205,
692.07268935,  185.112186  ,  689.86693017,  230.        ,
745.09017748,   90.        ,  716.22569106,  118.66819383,
694.08127225,  138.90738144,  744.03210444,  230.        ,
790.        ,   90.        ,  753.85197839,  118.69199967,
734.3154154 ,  142.60063778,  712.17099658,  162.8398254 ]

config.params['turbine_pos'] = numpy.reshape(turbine_pos_array, [-1, 2])

config.info()

rf = ReducedFunctional(config)
J = rf.j(turbine_pos_array)
print "J = ", J 

