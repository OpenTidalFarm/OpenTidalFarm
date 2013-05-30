SHELL=/bin/bash
SUBDIRS := $(wildcard */.)

.PHONY: $(SUBDIRS) all

all: $(SUBDIRS)

$(SUBDIRS):
	@$(MAKE) -s -C $@
