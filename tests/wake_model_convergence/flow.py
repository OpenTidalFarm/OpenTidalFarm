from opentidalfarm import *

# define a mesh and some flow models to use
mesh = "mesh.xml" # choose a mesh
V = VectorFunctionSpace(Mesh(mesh), "CG", 2)
flow = interpolate(Expression(("1.0+x[0]/1280.", "1.0+x[1]/640.")), V)

basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration(mesh, inflow_direction=[1.,0.])
config.params["automatic_scaling"] = False
config.set_site_dimensions(site_x_start, site_x_start + site_x,
                           site_y_start, site_y_start + site_y)

deploy_turbines(config, 1, 1)
wake = AnalyticalWake(config, flow, model_type="Jensen")
rf = ReducedFunctional(config, forward_model=wake)
m0 = rf.initial_control()

minconv = helpers.test_gradient_array(rf.j, rf.dj, m0, seed=.05)
descriptor = "The gradient taylor remainder test"
info_str = "flow"
if minconv < 1.9 or minconv > 2.1 or numpy.isnan(minconv) or numpy.isinf(minconv):
    info_red("%s failed for the %s" % (descriptor, info_str))
else:
    info_green("%s passed for the %s" % (descriptor, info_str))
