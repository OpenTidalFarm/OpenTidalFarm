#!/usr/bin/env python
# -*- coding: utf-8 -*-

# .. _scenario1:
#
# .. py:currentmodule:: opentidalfarm
#
# Tidal simulation in the Orkney island
# =====================================
#
#
# Introduction
# ************
#
# This example demonstrates how OpenTidalFarm can be used for simulating the
# tides in a realistic domain.

# This example requires some large data files, that must be downloaded
# separately by calling in the source code directory:
#
# .. code-block:: bash
#
#    git submodule init
#    git submodule update

# Implementation
# **************

# We begin with importing the OpenTidalFarm module.

from opentidalfarm import *

# We also need the datetime module for the tidal forcing.

import datetime

# Next we define the UTM zone and band, as we we will need it multiple times
# later on.

utm_zone = 30
utm_band = 'V'

# Next we create shallow water problem and attach the domain and boundary
# conditions

prob_params = SWProblem.default_parameters()

# We load the mesh in UTM coordinates, and boundary ids from file

domain = FileDomain("../data/meshes/orkney/orkney_utm.xml")
prob_params.domain = domain

# The the mesh and boundary ids can be visualised with

#plot(domain.facet_ids, interactive=True)

# Next we specify boundary conditions. We apply tidal boundary forcing, by using
# the :class:`TidalForcing` class.

eta_expr = TidalForcing(grid_file_name='../data/netcdf/gridES2008.nc',
                        data_file_name='../data/netcdf/hf.ES2008.nc',
                        ranges=((-4.0,0.0), (58.0,61.0)),
                        utm_zone=utm_zone,
                        utm_band=utm_band,
                        initial_time=datetime.datetime(2001, 9, 18, 0),
                        constituents=['Q1', 'O1', 'P1', 'K1', 'N2', 'M2', 'S2', 'K2'])

plot(eta_expr, mesh=domain.mesh)
interactive()
