import configuration 
import numpy
import ipopt 
import IPOptUtils
from reduced_functional import ReducedFunctional
from dolfin import *
from helpers import info, info_green, info_red, info_blue
set_log_level(ERROR)

basin_x = 640.
basin_y = 320.

# We set the perturbation_direction with a constant seed, so that it is consistent in a parallel environment.
config = configuration.ScenarioConfiguration("mesh.xml", inflow_direction = [1, 0])
# Switch of the automatic scaling, as does not support scaling of the friction term yet 
config.params['automatic_scaling'] = False
config.params['controls'] = ['turbine_friction']

# Place one turbine 
offset = 0.0
turbine_pos = [[basin_x/3 + offset, basin_y/2 + offset]] 
config.set_turbine_pos(turbine_pos)

model = ReducedFunctional(config, scaling_factor = 0.01)
m0 = 0.16913721*numpy.ones(len(model.initial_control()))

f = IPOptUtils.IPOptFunction()
# Overwrite the functional and gradient function with our implementation
f.objective= model.j 
f.gradient= model.dj 

nlp = ipopt.problem(len(m0), 
                    0, 
                    f, 
                    numpy.zeros(len(m0)), 
                    # Set the maximum friction value to 1.0
                    1.0*numpy.ones(len(m0)))
nlp.addOption('mu_strategy', 'adaptive')
nlp.addOption('tol', 1e-5)
nlp.addOption('print_level', 5)
nlp.addOption('check_derivatives_for_naninf', 'yes')
nlp.addOption('obj_scaling_factor', -1.0)
nlp.addOption('hessian_approximation', 'limited-memory')
nlp.addOption('max_iter', 20)

m, sinfo = nlp.solve(m0)
info_green(sinfo['status_msg'])
info_green("Solution of the primal variables: m=%s\n" % repr(m))
info_green("Solution of the dual variables: lambda=%s\n" % repr(sinfo['mult_g']))
info_green("Objective=%s\n" % repr(sinfo['obj_val']))

info_green("The optimal friction coefficient is m = " + str(m) + ".")
