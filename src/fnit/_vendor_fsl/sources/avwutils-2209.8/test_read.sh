#!/bin/sh

#   basic image handling scripts
#
#   FMRIB Image Analysis Group
#
#   Copyright (C) 1999-2004 University of Oxford
#
#   SHCOPYRIGHT

for fn in epimj?.* episs?.* ; do
   echo "  XXXXXXX $fn XXXXXXX "
  ./avwstats $fn -m -M -v -V -r 
  ./fslstats $fn -m -M -v -V -r 
done >& log_testread.txt

cat log_testread.txt


