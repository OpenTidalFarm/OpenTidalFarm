# Builds a Docker image with OpenTidalFarm master
# version built from gitsources. It is based on
# the dev-dolfin-adjoint image available at
#
#      quay.io/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Simon Funke <simon@simula.no>

FROM quay.io/dolfinadjoint/dolfin-adjoint:dolfin-adjoint-2017.2.0
MAINTAINER Simon Funke <simon@simula.no>

USER root
RUN pip install pyyaml && \
    pip install uptide==0.3 && \
    pip install utm && \
    pip install https://github.com/OpenTidalFarm/OpenTidalFarm/archive/master.zip
RUN git clone https://github.com/OpenTidalFarm/OpenTidalFarm.git
RUN cd OpenTidalFarm; git submodule init; git submodule update

RUN apt-get -qq update && \
    apt-get -y install gmsh

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME

USER root
