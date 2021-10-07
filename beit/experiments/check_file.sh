#!/bin/bash

set -x

for occ in none semi wd wm wl all nall
do
  for split in 'train' 'val' 'test'
  do
    tail -59 test_${occ}_${split}.py | head -5
  done
done
