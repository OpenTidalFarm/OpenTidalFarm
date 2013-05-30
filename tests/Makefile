SHELL=/bin/bash
SUBDIRS := $(wildcard */.)

all:
	-for d in $(SUBDIRS); do (cd $$d; make ); done
