from opentidalfarm import *
set_log_level(INFO)

# Some domain information extracted from the geo file
basin_x = 1280.
basin_y = 640.
site_x = 640.
site_y = 320.
mesh = 'mesh.xml'
bathymetry = 'bathymetry.xml'



class Bathymetry(Expression):
    def __init__(self, config, max_depth, min_depth):
        self.config = config
        self.step = (config.domain.site_y_start + config.domain.site_y_end)/3
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.m = (self.max_depth-self.min_depth)/self.step
        self.m2 = (self.min_depth- self.max_depth)/self.step

    def eval(self, values, x):
        if x[1] > 2 * self.step:
            values[:] = self.m2 * (x[1] - 2 * self.step) + self.max_depth + 0.5*sin(100 * x[0]) + sin(0.5 * x[0]) + 0.25*sin(50 * x[1]) + 0.5*sin(0.5 * x[1])
        elif x[1] > self.step and x[1] <= 2 * self.step:
            values[:] = self.max_depth +  0.5*sin(100 * x[0]) + sin(0.5 * x[0]) + 0.25*sin(50 * x[1]) + 0.5*sin(0.5 * x[1])
        else:
            values[:] = (self.m * x[1] + self.min_depth) + 0.5*sin(100 * x[0]) + sin(0.5 * x[0]) + 0.25*sin(50 * x[1]) + 0.5*sin(0.5 * x[1])

print "Setting up bathymetry..."
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration(mesh, inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

bathymetry = Bathymetry(config, 50., 40.)
depth = interpolate(bathymetry, FunctionSpace(config.domain.mesh, "CG", 1))
config.params["depth"] = depth
config.params["automatic_scaling"] = True 
config.params['ambient_flow_xml'] = 'ambient_flow.xml'
# Place some turbines 
config.params['ecology'] = True
config.params['bathymetry'] = True
config.params["print_individual_turbine_power"] = True
deploy_turbines(config, nx = 8, ny = 4, friction=21)

#config.params["depth_xml"] = bathymetry
#depth = interpolate(bathymetry, FunctionSpace(config.domain.mesh, "CG", 1))
#if config.params['bathymetry'] == True:
#      config.params['depth'] = depth #Function(config.function_space, config.params['depth_xml'])

   
config.info()

rf = ReducedFunctional(config)
m0 = rf.initial_control()
print "Functional value: ", rf(m0)

lb, ub = position_constraints(config) 
ineq = get_minimum_distance_constraint_func(config)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP", options = {"maxiter": 150})

################################
#from math import log
#import matplotlib.pyplot as plt

#def trans_vel(U1,Z,Z1,P):
#        U = U1*((Z/Z1)**P)
#        return U


#Za = float(input("Enter transition height:"))
#Z1 = float(input("Enter reference height: "))
#P= 1.0/5

#def frict_vel(U,Z,Z0,k):
#        z=Z/Z0
#        U0 = (U*k)/log(z)
#        return U0


#def base_vel(U0,Z,Z0,k):
#        z = Z/Z0
#        U = (U0/k)*log(z)
#        return U


#Zb = float(input("Enter height above seabed to calc to:"))
#Z0 = float(input("Enter roughness height: "))
#k = 0.4

######################################

#Xrange = range(0,640,1)
#Yrange = range(0,320,1)


#output = open("vel_mag.csv", 'w')
#output.write("x,y,z,velocity\n")

#for j in Yrange:
#        for i in Xrange:
             #   xvel = rf.last_state((i,j))[0]
            #    yvel = rf.last_state((i,j))[1]
           #     mag = (xvel**2 + yvel**2)**0.5
          #      U_ref = trans_vel(mag,Za,Z1,P)
         #       U0 = frict_vel(U_ref,Za,Z0,k)
        #        base_mag = base_vel(U0,Zb,Z0,k)
       #         output.write(str(i))
      #          output.write(",")
     #           output.write(str(j))
   #             output.write(",")
  #              output.write("1")
    #            output.write(",")
 #               output.write(str(base_mag))
#                output.write("\n")

#output.close()
