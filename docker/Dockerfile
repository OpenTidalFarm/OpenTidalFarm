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
RUN pip install pyyaml && \
    pip install git+git://github.com/stephankramer/uptide.git && \
    pip install utm && \
    pip install https://github.com/OpenTidalFarm/OpenTidalFarm/archive/master.zip
RUN git clone https://github.com/OpenTidalFarm/OpenTidalFarm.git

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME
RUN echo "export PYTHONPATH=~/OpenTidalFarm/:\$PYTHONPATH" > ~/.bashrc

USER root
