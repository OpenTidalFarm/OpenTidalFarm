.. OpenTidalFarm documentation master file, created by
   sphinx-quickstart on Thu Jul 17 14:18:42 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenTidalFarm's documentation!
==========================================

OpenTidalFarm is an open-source software for simulating and optimising tidal
turbine farms.

The positioning of the turbines in a tidal farm is a crucial decision.
Simulations show that the optimal positioning can increase the power generation
of the farm by up to 50% and can therefore determine the viability of a project.
However, finding the optimal layout is a difficult process due to the complex
flow interactions. OpenTidalFarm solves this problem by applying an efficient
optimisation algorithm onto a accurate flow prediction model.

Following presentation gives a quick introduction to OpenTidalFarm:
`OpenTidalFarm
<https://www.slideboom.com/presentations/758051/OpenTidalFarm/>`_

How to get started
------------------

1. Download and :doc:`install OpenTidalFarm <installation>`.
2. Try some of our :doc:`examples <examples>`.
3. Read the :doc:`programmers reference <reference>` to set up your own study.

For questions and to report issues use the `GitHub issue tracker
<https://github.com/OpenTidalFarm/OpenTidalFarm/issues>`_.


Features
--------

A selection of features are:

- nonlinear shallow water model for flow predictions;
- prediction of power production of a farm;
- an efficient adjoint model to compute sensitivities;
- optimise the turbine position and size to maximise the total farm power output;
- site constraints and minimum distance between turbines;
- optimisation of hundreds of turbines;
- parallel support using MPI;
- checkpoint support to restart optimisation.

Contents
========

.. toctree::
   :maxdepth: 2
   :numbered:

   installation
   examples
   reference
   contributing
   team
   citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Licence
=======

OpenTidalFarm is an open source project that can be freely used under the `GNU GPL version 3 licence`_.

.. _GNU GPL version 3 licence: http://www.gnu.org/licenses/gpl.html
