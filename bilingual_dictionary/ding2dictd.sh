#!/bin/bash
sourcefile=$1
targetfile=$2
gawk -f ding2dictd.awk $sourcefile|dictfmt -f \
-s "$1 extracted from en.wiktionary.org" -u "http://en.wiktionary.org/wiki/User:Matthias_Buchmeier" \
--utf8  --columns 0 --without-headword --headword-separator :: $targetfile