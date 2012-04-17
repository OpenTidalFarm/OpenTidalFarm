#!/bin/sh
for a in *.jpg; do convert -trim "$a" "$a"; done
