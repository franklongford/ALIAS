#!/bin/bash

declare  traj="$1"
declare  top="$2"
declare  recon="$3"
declare  ow_coeff="$4"
declare  ow_recon="$5"
declare  ow_intpos="$6"
declare  ow_hist="$7"


python src/main.py $traj $top $recon $ow_coeff $ow_recon $ow_intpos $ow_hist

