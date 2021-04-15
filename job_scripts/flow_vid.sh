#!/bin/bash

#SBATCH -n 1
#SBATCH -t 0:30:00
#SBATCH -p gpu_short

# # load modules
# module load python

# #Copy input file to scratch
# cp -r $HOME/code/datasets/flow_vid_separated/0 "$TMPDIR"
# cp -r $HOME/code/datasets/flow_vid_separated/generated_masks $TMPDIR

# #Create output directory on scratch
# mkdir "$TMPDIR"/output_dir

# #Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# echo "Start: $(date)" >> $HOME/job_logs/flow_vid.log
# python $HOME/code/complete_video.py --data_dir $TMPDIR --mask_dir $TMPDIR/generated_masks --video 0 --outroot $TMPDIR/output_dir
# echo "End: $(date)" >> $HOME/job_logs/flow_vid.log

# #Copy output directory from scratch to home
# cp -r "$TMPDIR"/output_dir $HOME/code/datasets/flow_vid_separated/completed


# for i in 0 1 2 3 4
# do
#     echo "Start $i: $(date)" >> $HOME/job_logs/flow_vid.log
#     python complete_video.py --data_dir results/flow_vid_separated --mask_dir results/flow_vid_separated/generated_masks --video $i --outroot results/flow_vid_separated/completed/$i
# done

# python flow_vid_things.py

python complete_video.py --data_dir results/flow_vid_separated --mask_dir results/flow_vid_separated/generated_masks --video 0 --outroot results/flow_vid_separated/completed/0
