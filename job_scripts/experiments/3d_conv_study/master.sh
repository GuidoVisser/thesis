#!/bin/bash

for EXPERIMENT in scooter-black_normal_3d_bottleneck kruispunt_rijks_normal_3d_bottleneck amsterdamse_brug_normal_3d_bottleneck scooter-black_3d_encoder kruispunt_rijks_3d_encoder amsterdamse_brug_3d_encoder scooter-black_fully 2d kruispunt_rijks_fully 2d amsterdamse_brug_fully 2d 
do
    sbatch job_scripts/experiments/3d_conv_study/$EXPERIMENT.sh
done
