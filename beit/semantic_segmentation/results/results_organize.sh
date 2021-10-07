#!/bin/bash

metric='mIoU'

for metric in mIoU mAcc oAcc
do
  echo ${metric}
  for occ in none semi wd wm wl nall all
  do
    read -r _ _ _ train_mIoU _ train_mAcc _ train_oAcc _ <<< `tail -5 ${occ}_train_results.txt | head -1`
    read -r _ _ _ val_mIoU _ val_mAcc _ val_oAcc _ <<< `tail -5 ${occ}_val_results.txt | head -1`
    read -r _ _ _ test_mIoU _ test_mAcc _ test_oAcc _ <<< `tail -5 ${occ}_test_results.txt | head -1`
  
    train_results=train_${metric}
    val_results=val_${metric}
    test_results=test_${metric}

    echo ${!train_results} ${!val_results} ${!test_results}
  done
  echo ''
done
