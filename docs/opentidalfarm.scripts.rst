Useful scripts
==============

The ``scripts`` directory of the OpenTidalFarm source directory contains a set
of useful scripts.


FVCOM to OpenTidalFarm converter
--------------------------------

This script can be used to convert an existing FVCOM mesh into a compatible
OpenTidalFarm mesh. It can also (optionally) convert FVCOM velocity fields.

Usage:

.. code-block:: bash

    usage: fvcom_to_otf.py [-h] --nc NC --xml XML [--velocity VELOCITY] [--plot]

    Converts FVCOM meshes and velocity fields to OpenTidalFarm compatible xml
    files

    optional arguments:
      -h, --help           show this help message and exit
      --nc NC              input FVCOM filename (.nc extension)
      --xml XML            output OpenTidalFarm mesh filename (.xml extension)
      --velocity VELOCITY  output OpenTidalFarm velocity filename (.xml extension)
      --plot               plot the results

Example:

.. code-block:: bash

    python scripts/fvcom_to_otf.py --nc myFCVOM.nc --xml myOTFmesh.xml --velocity myOTF_velocities.xml --plot
