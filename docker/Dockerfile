# Builds a Docker image with OpenTidalFarm master
# version built from gitsources. It is based on
# the dev-dolfin-adjoint image available at
#
#      quay.io/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Simon Funke <simon@simula.no>

FROM quay.io/dolfinadjoint/dev-dolfin-adjoint:latest
MAINTAINER Simon Funke <simon@simula.no>

USER root
RUN pip2 install pyyaml && \
    pip2 install git+git://github.com/stephankramer/uptide.git && \
    pip2 install utm && \
    pip2 install https://github.com/OpenTidalFarm/OpenTidalFarm/archive/master.zip
RUN git clone https://github.com/OpenTidalFarm/OpenTidalFarm.git
RUN cd OpenTidalFarm; git submodule init; git submodule update

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME

USER root
