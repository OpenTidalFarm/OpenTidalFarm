from opentidalfarm import *
set_log_level(PROGRESS)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
eta0 = (2.0+1e-10)/sqrt(9.81/50) # This will give a inflow velocity of 2m/s
config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1,0], period = 10.*60, eta0=eta0)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['initial_condition'] = ConstantFlowInitialCondition(config)
config.params['automatic_scaling'] = False

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
turbine_pos_array = [ 170.00000071,   90.00000415,  170.0000035 ,  213.82194722,
	170.00000163,  106.01276118,  170.00000149,  229.99999368,
	169.99999988,  167.40086748,  170.00000364,  198.22732824,
	170.0000014 ,  121.54082911,  170.00000167,  137.02257782,
	197.13694339,  132.54629338,  169.99999905,  152.33845909,
	170.00000023,  182.73718427,  196.60485632,  189.36761746,
	198.75508648,   90.00000206,  195.0645479 ,  170.58133964,
	195.21600967,  151.43474366,  198.84281199,  230.00000069,
	220.95188027,   90.00000044,  200.48712505,  112.80192672,
	199.27540535,  208.71968744,  221.0988914 ,  230.00000125,
	263.02363535,   89.99999766,  356.55341176,  159.0678451 ,
	357.237474  ,  180.90876751,  266.98305229,  229.99999896,
	348.64630837,   89.9999983 ,  356.62579248,  137.336928  ,
	358.88672295,  203.22856444,  357.16535723,  229.99999947,
	375.0598651 ,   89.99999874,  357.37150133,  115.59653878,
	422.75805785,   90.00052496,  398.81872343,  229.99999924]
turbine_pos = numpy.reshape(turbine_pos_array, [-1, 2])
turbine_pos_mirror = []
for x, y in turbine_pos:
	turbine_pos_mirror.append([-(x-(160+160))+(160+160), y])

turbine_pos_mirror_array = numpy.reshape(turbine_pos_mirror, -1)

config.params['turbine_pos'] = turbine_pos_mirror 

config.info()

rf = ReducedFunctional(config)
J = rf.j(turbine_pos_mirror_array)
print "J = ", J 

