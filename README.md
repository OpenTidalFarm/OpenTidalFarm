---
layout: default
---

About OpenTidalFarm 
===================
OpenTidalFarm is a layout optimisation software for tidal turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision.
Simulations show that the optimal positioning can increase the power generation of the farm by up to 50%  
and can therefore determine the viability of a project.
 
However, finding the optimal layout is a difficult process due to the complex flow interactions.
OpenTidalFarm solves this problem by applying an efficient optimisation algorithm onto a accurate 
flow prediction model.

OpenTidalFarm is open source software and can be downloaded and used for free.

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
<iframe class="youtube-player" type="text/html" width="640" height="385" src="http://www.youtube.com/embed/ng3bbso-vGk" frameborder="0">
</iframe>

Getting started
===============
Note: This installation procedure assumes that you are running the [Ubuntu](http://www.ubuntu.com/) operating system.

The installation consists of following steps

1. Download and install the dependencies: [FEniCS project](http://fenicsproject.org/download/) and [dolfin-adjoint](http://dolfin-adjoint.org/download/index.html)
2. [Download OpenTidalFarm](https://github.com/funsim/OpenTidalFarm/zipball/master) and extract it.
3. Open a terminal and change into the extracted directory. Then run
`source export_path.sh`
to include the OpenTidalFarm to you system path.

Now you are ready to run one of the many examples in the `examples/` folder.

<!--Contribution
============
If you are interested in contributing to the project, please send me an email and create a fork on github. 
-->

<!--
Documentation
=============
-->

Contact 
=======
<a id="contact"> </a>
For questions and support please contact Simon W. Funke <s.funke09@imperial.ac.uk>.

Licence
=======
OpenTidalFarm is an open source project that can be freely used under the 
[GNU GPL version 3](http://www.gnu.org/licenses/gpl.html)
licence.
