from opentidalfarm import *
import os
set_log_level(INFO)
"""
This example demsonstrates the basinhopping algorithm. 4 Turbines are deployed
in an small area 30m deep. A larger area 20m deep is also within the turbine
site. The water is 40m deep elsewhere. When using a local minimizer the turbines
will not 'hop' across the deeper water to the even shallower water and will
optimize in the area 30m deep. With basinhopping the turbines will eventually
jump the gap to the faster flowing water in the area that is 20m deep.

 /----------------------------------------------------------------------------\
 |                                      *                                     | 
 |                                      *                                     | 
 |             *******                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             * 30m *       40m        *                  20m                |    
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *     *                  *                                     | 
 |             *******                  *                                     | 
 |                                      *                                     | 
 |                                      *                                     | 
 \----------------------------------------------------------------------------/
"""

# Some domain information extracted from the geo file
basin_x = 640.
basin_y = 320.
site_x = 320.
site_y = 160.
site_x_start = (basin_x - site_x)/2 
site_y_start = (basin_y - site_y)/2 
config = SteadyConfiguration("mesh.xml", inflow_direction = [1, 0])
config.set_site_dimensions(site_x_start, site_x_start + site_x, \
                           site_y_start, site_y_start + site_y) 

# Define some parameters for the isolated area
turbine_size = config.params["turbine_x"]
min_x = 220.
max_x = 270.
min_y = 80.
max_y = 240.
start_y = min_y+0.5*turbine_size
end_y = max_y-0.5*turbine_size
diff_y = end_y-start_y
turbines = 4
spacing_y = diff_y/float(turbines-1)

# depths of the three areas
depth_isolated = 30.
depth_shallow = 20.
depth_deep = 40.

# Place some turbines in the isolated area
positions = []
for i in range(turbines):
  positions.append( (((max_x+min_x)/2.), start_y+(i*spacing_y)) )

config.set_turbine_pos(positions)
config.info()

# Define an expression for the bathymetry in a C++ snippet
cppcode=("""
class Conditional: public Expression{
  public:
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    double depth_isolated;
    double depth_shallow;
    double depth_deep;

    void eval(dolfin::Array<double>& values, const dolfin::Array<double>& x) const{
      // for x less than 320m define an area 30m deep
      if (x[0] < 320.0){
        if ((x[0] > min_x && x[0] < max_x) && (x[1] > min_y && x[1] < max_y)){
          values[0] = depth_isolated;
        }
        // if its not the isoated area and x < 320 the depth should be 40m
        else{
          values[0] = depth_deep;
        }
      }
      // if x > 320m the depth should be 20m
      else{
        values[0] = depth_shallow;
      }     
    }
};
""")

# Set the bathymetry to that defined above 
bathymetry = Expression(cppcode)
bathymetry.min_x = min_x
bathymetry.min_y = min_y
bathymetry.max_x = max_x
bathymetry.max_y = max_y
bathymetry.depth_isolated = depth_isolated
bathymetry.depth_shallow = depth_shallow
bathymetry.depth_deep = depth_deep

# Interpolate the depth onto the function space of the domain and save to a pvd
depth = interpolate(bathymetry, FunctionSpace(config.domain.mesh, "CG", 1))
depth_pvd = File("bathymetry.pvd")
depth_pvd << depth

# Set the depth in the config
config.params["depth"] = depth

rf = ReducedFunctional(config)
lb, ub = position_constraints(config) 

# Solve the problem using L-BFGS-B to show the turbines do not hop to the 
# shallower area where water is faster flowing.
maximize(rf, method = "L-BFGS-B", bounds = [lb, ub])

# Move the output files to a new directory
path = "./L-BFGS-B/"
if MPI.process_number()==0:
  if not os.path.exists(path):
    os.makedirs(path)
  
  for filename in os.listdir("."):
    if filename.startswith("turbine")\
    or filename.startswith("power")\
    or filename.startswith("iter")\
    or filename.startswith("functional"):
      os.rename(filename, path+filename)
