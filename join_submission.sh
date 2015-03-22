#!/bin/bash

# This script must be launch AFTER the main.py. 
# It concatenate our submissions (max 10) equivalent to the number of CPU used.

test ! -f submission_all.csv || exit 0

echo "driver_trip,prob" > submission_all.csv

for file in `ls submission_[0-9].csv`
do
  cat $file >> submission_all.csv
done 

mv submission_[0-9].csv /tmp/

