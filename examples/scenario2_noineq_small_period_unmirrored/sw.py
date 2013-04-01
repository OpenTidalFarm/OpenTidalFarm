from opentidalfarm import *
set_log_level(PROGRESS)

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = UnsteadyConfiguration("mesh.xml", inflow_direction = [1,0], period = 10.*60)
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)
config.params['automatic_scaling'] = False

# Place some turbines 
deploy_turbines(config, nx = 8, ny = 4)
turbine_pos_array = [ 169.9999999 ,   89.99999981,  170.42874408,  107.03999637,
						170.47028971,  212.96347159,  169.9999999 ,  230.00000018,
						169.99999966,  182.47622214,  169.99999972,  197.46483429,
						169.99999986,  122.55902233,  169.99999979,  137.56902967,
						188.81618266,   89.99999988,  169.99999975,  152.59802621,
						169.99999959,  167.45632917,  188.90939586,  230.00000011,
						229.10563281,   89.99999996,  236.69346104,  120.81624999,
						236.64298835,  197.03544291,  229.05847735,  230.00000004,
						257.19724405,   89.99999995,  235.13848891,  171.0468504 ,
						235.15183876,  145.38544254,  258.82470338,  230.00000005,
						305.322531  ,   89.99999998,  310.61797203,  152.29482055,
						313.0645626 ,  207.32550882,  307.34195518,  230.00000003,
						328.80569483,   89.99999997,  311.24345427,  133.38483608,
						310.59242822,  170.63688308,  330.86993196,  230.00000003,
						370.94484682,   89.99999998,  313.07690025,  114.34854002,
						311.21701781,  188.83356058,  378.51678395,  230.00000001]

config.params['turbine_pos'] = numpy.reshape(turbine_pos_array, [-1, 2])

config.info()

rf = ReducedFunctional(config)
J = rf.j(turbine_pos_array)
print "J = ", J 

