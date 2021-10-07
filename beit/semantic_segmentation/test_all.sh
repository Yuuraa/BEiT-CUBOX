#!/bin/bash

set -x

for occ in all nall none semi wd wm wl
do
  for split in test val train
  do
    bash tools/dist_test.sh ../experiments/test_${occ}_${split}.py ../experiments/iter_160000.pth 4 --eval mIoU > results/${occ}_${split}_results.txt
  done
done
