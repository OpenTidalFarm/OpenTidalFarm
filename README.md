About OpenTidalFarm 
===================
OpenTidalFarm is a layout optimisation software for tidal turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision. Simulations show that the optimal positioning can increase the power generation of the farm by up to 50% and can therefore determine the viability of a project.
 
However, finding the optimal layout is a difficult process due to the complex flow interactions. OpenTidalFarm solves this problem by applying an efficient optimisation algorithm onto a accurate flow prediction model.

Following presentation gives a quick introduction to OpenTidalFarm:
<div style="width:425px;text-align:left"><a style="font:14px Helvetica,Arial,Sans-serif;color: #0000CC;display:block;margin:12px 0 3px 0;text-decoration:underline;" href="//www.slideboom.com/presentations/758051/OpenTidalFarm" title="OpenTidalFarm">OpenTidalFarm</a><object classid="clsid:d27cdb6e-ae6d-11cf-96b8-444553540000" codebase="http://fpdownload.macromedia.com/pub/shockwave/cabs/flash/swflash.cab#version=9,0,28,0" width="425" height="370" id="onlinePlayer758051"><param name="movie" value="//www.slideboom.com/player/player.swf?id_resource=758051" /><param name="allowScriptAccess" value="always" /><param name="quality" value="high" /><param name="bgcolor" value="#ffffff" /><param name="allowFullScreen" value="true" /><param name="flashVars" value="" /><embed src="//www.slideboom.com/player/player.swf?id_resource=758051" width="425" height="370" name="onlinePlayer758051" type="application/x-shockwave-flash" pluginspage="http://www.macromedia.com/go/getflashplayer"allowScriptAccess="always" quality="high" bgcolor="#ffffff" allowFullScreen="true" flashVars="" ></embed></object>
</div>

Examples
========
The following video demonstrates OpenTidalFarm on different setups:

* [Optimisation of 32 turbines in an idealised tidal stream](http://www.youtube.com/embed/ng3bbso-vGk?vq=hd1080)

The following videos show the how the turbine positions change during the optimisation. The final frame of the videos show 
the optimisation locations of the turbines. In these cases, the total power production of the turbines was improved by over 25%.  
* [Optimisation of 128 turbines in the Pentland Firth, Scotland](http://www.youtube.com/embed/mMNes2Ubz2Y?vq=hd1080)
* [Optimisation of 128 turbines in the Pentland Firth, Scotland (zoomed into site area)](http://www.youtube.com/embed/GjWNBzvSeSs?vq=hd1080)
* [Optimisation of 256 turbines in the Pentland Firth, Scotland](http://www.youtube.com/embed/3yOeCL5_Vrs?vq=hd1080)
* [Optimisation of 256 turbines in the Pentland Firth, Scotland (zoomed into site area)](http://www.youtube.com/embed/p6U5_7Su58E?vq=hd1080)

Features 
========
* High resolution shallow water model for accurate flow prediction.
* Arbitrary shoreline data and bathymetry support.
* Optimise the turbine position and size to maximise the total farm power output.
* Site constraints / minimum distance between turbines.
* Optimisation of up to hundreds of turbines.
* Checkpoint support to restart optimisation.
 
Installation
============
Note: This installation procedure assumes that you are running the [Ubuntu](http://www.ubuntu.com/) operating system.

The installation consists of following steps

1. Download and install the dependencies:
    - [FEniCS project](http://fenicsproject.org/download/) 
    - [dolfin-adjoint](http://dolfin-adjoint.org/download/index.html).
    - [SciPy Version >0.11](https://github.com/scipy/scipy).
2. [Download OpenTidalFarm](https://github.com/funsim/OpenTidalFarm/zipball/master) and extract it.
3. Open a terminal and change into the extracted directory and run

```bash
sudo python setup.py install
```

to install it. A simple test to check if the installation was correct is to open a Python shell and type:

```python
from opentidalfarm import *
```

If you get an error, make sure that you have set the `PYTHONPATH` correctly. In Linux, this can be done with:

```bash
export PYTHONPATH=/XYZ:$PYTHONPATH
```
where `XYZ` should be replaced with the path to your OpenTidalFarm installation. 

Now you are ready to run one of the many examples in the `examples/` folder.



Getting started 
========================
Following example code shows how to optimise the position of 32 turbines in a mesh of the Orkney islands.
```python
from opentidalfarm import *

config = SteadyConfiguration("mesh/earth_orkney_converted_coarse.xml", inflow_direction=[0.9865837220518425, -0.16325611591095968])
config.params['diffusion_coef'] = 90.0
config.params['turbine_x'] = 40.
config.params['turbine_y'] = 40.
config.params['controls'] = ['turbine_pos']

# Some domain information extracted from the geo file.
# This information is used to deploy the turbines autmatically.
site_x = 1000.
site_y = 500.
site_x_start = 1.03068e+07
site_y_start = 6.52246e+06 - site_y
config.set_site_dimensions(site_x_start, site_x_start + site_x, site_y_start, site_y_start + site_y)

# Place 32 turbines in a regular grid, each with a maximum friction coefficient of 10.5
deploy_turbines(config, nx = 8, ny = 4, friction=10.5)

# Define some constraints for the optimisation positions.
# Constraint to keep the turbines within the site area 
lb, ub = position_constraints(config)
# Constraint to keep a minium distance of 1.5 turbine diameter between each turbine
ineq = get_minimum_distance_constraint_func(config)

# Solve the optimisation problem
rf = ReducedFunctional(config, plot = True)
maximize(rf, bounds = [lb, ub], constraints = ineq, method = "SLSQP")
```

This example can be found in the `examples/tutorial` directory and can be executed by running `make mesh && make`.

The output files are:
* turbine.pvd: The turbine positions at each optimisation step
* p2p1_u.pvd: The velocity function for the most recent turbine position calculation. 
* p2p1_p.pvd: The free-surface displacement function for the most recent turbine position calculation.

Documentation
=============
## Configurations ##
OpenTidalFarm is based on configurations for defining different setups.
The most important configurations are
* `SteadyConfiguration`: Use this configuration for steady simulations.
* `UnstadyConfiguration`: Use this configuration for unsteady simulations.

Once you have a configuration object, try running
```python
config.info()
```
to see detailed information about the settings of the current configuration. 

### Parameters ###
Each configuration has a large of additional parameters that can be changed.

For example to use Picard iterations instead of a Newton solver to solve the nonlinear shallow water equations one would set:
```python
config.params["newton_solver"] = False 
```

Some of the more important parameters are:
* "controls": Defines the control parameters that the optimisation algorithm may use. Possible choicees are the optimisation of the turbine positions and/or the friction of each individual turbine. Valid values: a list containing one or more of `['turbine_pos', 'turbine_friction']`.
* "save_checkpoints": Automatically save checkpoints to disk from which the optimisation can be restarted. Valid values: `True` or `False`.
* "turbine_x": The x-extension of each turbine.
* "turbine_y": The y-extension of each turbine.
* "output_turbine_power": Output the energy production for each individual turbine. Valid values: `True` or `False`.

Again, use `config.info()` to list the current configuration setup.

## Positioning constraints ##
Often, one ones to ensure that the turbines must be deployed in a particular area. 
This can be achieved in two ways:

### Box constraints ###
Box constraints is the easiest solution of the deployment area is a rectangle. 

The code below shows an example usage:
```python
config.set_site_dimensions(site_x_start, site_x_end, site_y_start, site_y_end)
bounds = position_constraints(config)
maximize(rf, bounds=bounds, method = "SLSQP")
```

### Inequality constraints for polygon shaped site domains ###
For more complex site domains, one can define a polygon in which the turbines must be deployed.

The code below shows an example usage for a domain polygon with 4 vertices:
```python
vertices = [[site_x_start, site_y_start], 
            [site_x_end,   site_y_start], 
            [site_x_end,   site_y_end], 
            [site_x_start, site_y_end]]

ineq = generate_site_constraints(config, vertices)
maximize(rf, constraints=ineq, method = "SLSQP")
```

Note that SLSQP in an infeasible algorithm, which means that the initial turbine layout does not have to fulfill these constraints.



## Mesh boundary IDs ##
OpenTidalFarm expects the 3 the mesh to have three identifiers of the boundary mesh:
 * ID 1: inflow boundary
 * ID 2: outflow boundary 
 * ID 3: shoreline boundary

## Optimisation options ##
The available optimisation options are explained in detail [here](http://dolfin-adjoint.org/documentation/optimisation.html).

## Advanced options ##

### Checkpointing ###
#### Creating checkpoints ####
OpenTidalFarm can automatically store checkpoints to disk from which the optimisation procedure can be restarted.
The checkpoints generation is activated with: 
```python
config.params["save_checkpoints"] = True
```
where `config` is the `Configuration` object.
The checkpoint data is stored in the two files "checkpoint_fwd.dat" and "checkpoint_adj.dat".

#### Loading from checkpoint ####
In order to restart from a checkpoint you need to load in the checkpoint with:
```python
rf.load_checkpoint() 
```
where `rf` is the `ReducedFunctionalObject`.

You will see that the optimisation starts from the beginning, however the optimisation iterations 
until the checkpoint will happen instantly since the solutions are cached in the checkpoint. 

### Compiler optimisations ###
By default, OpenTidalFarm only uses the `-O3` compiler optimisation flag as a safe choice.

However, for large optimisations runs, one might want to use more aggressive compiler optimisation.
Experiments have shown that following options can yield significant speed up:
```python
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
```
or the less aggressive version:
```python
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -fno-math-errno -march=native'
```
Add this line just before you call the maximize function. 

However, note that in some circumstances such aggressive optimisation might be problematic for the optimisation algorithms. If the optimisation algorithm returns errors saying that the gradient 

## Frequently asked questions ##
### Optimisation stops with Exit mode 8 ###
Sometimes, the SLSQP optimisation algorithm stops early giving the error:
```
Positive directional derivative for linesearch    (Exit mode 8)
```
In such case you can try following things:
* If you have the `-ffast-math -march=native` compiler flags active (see above), try switching them off. 
* Use finer mesh in the turbine site area. The numerical errors of representing the turbines might be dominating the problem.
* Use a looser optimisation tolerance, by passing the `tol` parameter to maximize function. 
* If you are only optimising for the turbine friction and you do not use `automatic_scaling` parameter (which you should in this particular case), then you might have to scale your problem manually. This is done by passing a scale argument to maximize, e.g. `maximize(rf, scale=1e-3)`. Choose the scale value such that the first gradient is in the order of 1. 

Citing
======
Please cite the following paper if you are using OpenTidalFarm:

S.W. Funke, P.E. Farrell, M.D. Piggott, Tidal turbine array optimisation using the adjoint approach, Renewable Energy, submitted (2013) [arXiv:1304.1768](http://arxiv.org/abs/1304.1768)

For the automated optimisation framework used by OpenTidalFarm, please cite:

Simon W. Funke and Patrick E. Farrell. A framework for automated PDE-constrained optimisation, TOMS, submitted. [arXiv:1302.3894](http://arxiv.org/abs/1302.3894)

For the automatic adjoint generation used by OpenTidalFarm, please cite:

Patrick E. Farrell, David A. Ham, Simon W. Funke and Marie E. Rognes (2012). Automated derivation of the adjoint of high-level transient finite element programs, accepted. [arXiv:1204.5577](http://arxiv.org/abs/1204.5577)

Contact 
=======
<a id="contact"> </a>
For questions and support please contact Simon Funke <s.funke09@imperial.ac.uk>.

Licence
=======
OpenTidalFarm is an open source project that can be freely used under the 
[GNU GPL version 3](http://www.gnu.org/licenses/gpl.html)
licence.
