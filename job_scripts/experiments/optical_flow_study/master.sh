#!/bin/bash

for EXPERIMENT in scooter-black_no_flow kruispunt_rijks_no_flow amsterdamse_brug_no_flow scooter-black_flow_rolloff kruispunt_rijks_flow_rolloff amsterdamse_brug_flow_rolloff 
do
    sbatch job_scripts/experiments/optical_flow_study/$EXPERIMENT.sh
done
