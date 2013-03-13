About OpenTidalFarm 
===================
OpenTidalFarm is a layout optimisation software for tidal turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision. Simulations show that the optimal positioning can increase the power generation of the farm by up to 50% and can therefore determine the viability of a project.
 
However, finding the optimal layout is a difficult process due to the complex flow interactions. OpenTidalFarm solves this problem by applying an efficient optimisation algorithm onto a accurate flow prediction model.

Features 
========
* High resolution shallow water model for accurate flow prediction
* Arbitrary shoreline data
* Optimization for power output
* Site constraints / minimum distance between turbines
* Up to hundreds of turbines

For additional features such as bathymetry support, enforcing a minimum/maximum turbine installation depth and cable costs please [contact me](#contact). 
 
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

       sudo python setup.py install

   to install it.

Now you are ready to run one of the many examples in the `examples/` folder.

Getting started
===============

Documentation
=============

Contact 
=======
<a id="contact"> </a>
For questions and support please contact Simon W. Funke <s.funke09@imperial.ac.uk>.

Licence
=======
OpenTidalFarm is an open source project that can be freely used under the 
[GNU GPL version 3](http://www.gnu.org/licenses/gpl.html)
licence.
