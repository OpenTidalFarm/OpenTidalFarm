SHELL=/bin/bash
SUBDIRS := $(wildcard */.)

.PHONY: $(SUBDIRS) all

all: $(SUBDIRS)
	@echo "All tests passed"

$(SUBDIRS):
	@$(MAKE) -s -C $@
