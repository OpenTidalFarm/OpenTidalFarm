About OpenTidalFarm 
===================
OpenTidalFarm is a layout optimisation software for tidal turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision. Simulations show that the optimal positioning can increase the power generation of the farm by up to 50% and can therefore determine the viability of a project.
 
However, finding the optimal layout is a difficult process due to the complex flow interactions. OpenTidalFarm solves this problem by applying an efficient optimisation algorithm onto a accurate flow prediction model.

Features 
========
* High resolution shallow water model for accurate flow prediction.
* Arbitrary shoreline data and bathymetry support.
* Optimise the turbine position and size to maximise the total farm power output.
* Site constraints / minimum distance between turbines.
* Optimisation of up to hundreds of turbines.
* Checkpoint support to restart optimisation.
 
Example
========
The following video demonstrates how OpenTidalFarm optimises 32 turbines in an idealised tidal stream.
[Youtube link](http://www.youtube.com/embed/ng3bbso-vGk)
</iframe>

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

   to install it.

Now you are ready to run one of the many examples in the `examples/` folder.

Getting started tutorial
========================

Documentation
=============

## Mesh boundary IDs ##
OpenTidalFarm expects the 3 the mesh to have three identifiers of the boundary mesh:
 * ID 1: inflow boundary
 * ID 2: outflow boundary 
 * ID 3: shoreline boundary

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
until the checkpoint will happen instantely since the solutions are cachced in the checkpoint. 

### Compiler optimisations ###
By default, OpenTidalFarm only uses the `-O3` compiler optimisation flag as a safe choice.

However, for large optimisations runs, one might want to use more aggressive compiler optimisation.
Experiments have shown that following options can yield significant speed up:
```python
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
```
Add this line just before you call the maximize function. 

However, be carefule that in some circumstances such aggressive optimisation might be problematic for the optimisation algorithms. If the optimisation algorithm returns errors saying that the gradient 

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

Contact 
=======
<a id="contact"> </a>
For questions and support please contact Simon W. Funke <s.funke09@imperial.ac.uk>.

Licence
=======
OpenTidalFarm is an open source project that can be freely used under the 
[GNU GPL version 3](http://www.gnu.org/licenses/gpl.html)
licence.
