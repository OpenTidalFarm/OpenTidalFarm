.. OpenTidalFarm documentation master file, created by
   sphinx-quickstart on Thu Jul 17 14:18:42 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenTidalFarm's documentation!
==========================================

OpenTidalFarm is a layout optimisation software for tidal turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision. Simulations show that the optimal positioning can increase the power generation of the farm by up to 50% and can therefore determine the viability of a project.

However, finding the optimal layout is a difficult process due to the complex flow interactions. OpenTidalFarm solves this problem by applying an efficient optimisation algorithm onto a accurate flow prediction model.

Following presentation gives a quick introduction to OpenTidalFarm:
`OpenTidalFarm
<https://www.slideboom.com/presentations/758051/OpenTidalFarm/>`_

To download the source code or to report issues visit the `GitHub page
<https://github.com/OpenTidalFarm/OpenTidalFarm>`_

Contents:

.. toctree::
   :maxdepth: 2
   :numbered:

   examples.rst
   features.rst
   installation.rst
   getting_started.rst
   contributing.rst

Code documentation
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


======
Citing
======

Please cite the following paper if you are using OpenTidalFarm:

S.W. Funke, P.E. Farrell, M.D. Piggott, Tidal turbine array optimisation using the adjoint approach, Renewable Energy, accepted (2013) `arXiv:1304.1768
<http://arxiv.org/abs/1304.1768>`_

For the automated optimisation framework used by OpenTidalFarm, please cite:

Simon W. Funke and Patrick E. Farrell. A framework for automated PDE-constrained optimisation, TOMS, submitted. `arXiv:1302.3894
<http://arxiv.org/abs/1302.3894>`_

For the automatic adjoint generation used by OpenTidalFarm, please cite:

Patrick E. Farrell, David A. Ham, Simon W. Funke and Marie E. Rognes (2013). Automated derivation of the adjoint of high-level transient finite element programs. SIAM Journal on Scientific Computing, Vol:35, ISSN:1064-8275, Pages:C369-C393

Developer team and contribution
===============================

OpenTidalFarm has been developed by 

* Simon Funke, Simula Research Laboratory
* Patrick Farrell, Oxford University
* Matthew Piggott, Imperial College London
* Stephan Kramer, Imperial College London
* David Culley, Imperial College London
* George Barnett, Imperial College London

If you would like to contribute to the project, please send an email to simon@simula.no.


Licence
=======

OpenTidalFarm is an open source project that can be freely used under the `GNU GPL version 3 licence`_.

.. _GNU GPL version 3 licence: http://www.gnu.org/licenses/gpl.html

