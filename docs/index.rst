.. raw:: html

    <link rel="stylesheet" href="_static/flexslider/flexslider.css" type="text/css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.6.2/jquery.min.js"></script>
    <script src="_static/flexslider/jquery.flexslider-min.js"></script>
    <script type="text/javascript" charset="utf-8">
      $(window).load(function() {
          $('.flexslider').flexslider();
      });
    </script>


Welcome to OpenTidalFarm
========================

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


.. raw:: html

    <div class="flexslider">
      <ul class="slides">
        <li>
          <img src="_static/slider_media/discrete_satellite.png" />
          <p class="flex-caption">OpenTidalFarm applied to the Pentland Firth</p>
        </li>
        <li>
          <img src="_static/slider_media/discrete_streamlines.png" />
          <p class="flex-caption">256 optimised tidal turbine locations with streamlines</p>
        </li>
        <li>
          <img src="_static/slider_media/discrete_turbine133_iter_plot.png" />
          <p class="flex-caption">Zoom in to optimised turbine locations</p>
        </li>
        <li>
          <img src="_static/slider_media/smooth_turbine.png" />
          <p class="flex-caption">Multi-farm tidal optimisation</p>
        </li>
        <li>
          <img src="_static/slider_media/otf_graphic.png" />
          <p class="flex-caption">The dynamical core of OpenTidalFarm. It implements a shallow water solver, an adjoint solver, and an optimisation loop.</p>
        </li>
      </ul>
    </div>

How to get started
------------------

1. Download and :doc:`install OpenTidalFarm <installation>`.
2. Try some of our :doc:`examples <examples>`.
3. Read the :doc:`programmers reference <reference>` to set up your own study.

For questions and to report issues use the `GitHub issue tracker
<https://github.com/OpenTidalFarm/OpenTidalFarm/issues>`_.

News
----

- 19.01.2015: OpenTidalFarm 1.5 released (compatibled with FEniCS 1.5)

  - Support of continuous farm optimisation.
  - Support for sensitivity analysis.

- 20.08.2014: OpenTidalFarm 1.4 released (compatibled with FEniCS 1.4)

  - Complete rewrite of OpenTidalFarm.

- 20.08.2014: OpenTidalFarm 0.9.1 released (compatibled with FEniCS 1.4)

  - Bugfix release.

- 16.07.2014: OpenTidalFarm 0.9 released (compatibled with FEniCS 1.4)

  - Initial release.
  - Support for discrete farm optimisation.


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
