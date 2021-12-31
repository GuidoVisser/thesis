import json
from easydict import EasyDict
from argparse import ArgumentParser
from os import makedirs, path
from math import ceil

def get_time_estimate(sec_per_epoch, n_epochs):
    return ceil(sec_per_epoch * n_epochs / 3600 * 1.25) + 2


def generate_script(config, video, identifier):

    time_estimate = f"{get_time_estimate(config.sec_per_epoch[video.name], config.n_epochs)}:00:00"
    print(time_estimate)

    header = f"""#!/bin/bash \n\n#SBATCH -n 1\n#SBATCH -t {time_estimate}\n#SBATCH -p gpu\n#SBATCH --gpus-per-node=gtx1080ti:4\n\n# get start time of script\nDT=`date +"%m_%d_%H_%M_%S"`\n\n# load modules\nmodule load 2020\nmodule load Python/3.8.2-GCCcore-9.3.0\n\n# install dependencies\npip install --user --upgrade torch && pip install --user --upgrade torchvision\n\n"""
    footer = f"""\n\necho "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log\n\n#Copy output directory from scratch to home\nmkdir -p $HOME/thesis/results/layer_decomposition/$VIDEO__$SLURM_JOBID__{identifier}\ncp -RT $TMPDIR/output_dir $HOME/thesis/results/layer_decomposition/$VIDEO__$SLURM_JOBID__{identifier}\n\nread -r t<$TMPDIR/output_dir/time.txt\necho $SLURM_JOBID $VIDEO {identifier} $t >> $HOME/thesis/times.txt"""

    mask_copying="\n".join([f"mkdir $TMPDIR/{i:02}\ncp -RT $HOME/thesis/datasets/{video.dataset}/Annotations/$VIDEO/{i:02}/{mask_path} $TMPDIR/{i:02}/{mask_path}" for i, mask_path in enumerate(video.mask_paths)])
    mask_argument=" ".join([f"$TMPDIR/{i:02}/{mask_path}" for i, mask_path in enumerate(video.mask_paths)])

    body = f"""#Copy input file to scratch\nVIDEO='{video.name}'\ncp -RT $HOME/thesis/datasets/{video.dataset}/Images/$VIDEO $TMPDIR/video\n{mask_copying}\nmkdir $TMPDIR/weights\ncp $HOME/thesis/models/third_party/weights/topkstm.pth $TMPDIR/weights/propagation_model.pth\ncp $HOME/thesis/models/third_party/weights/raft.pth $TMPDIR/weights/flow_model.pth\ncp $HOME/thesis/models/third_party/weights/monodepth.pth $TMPDIR/weights/depth_model.pth\n\n#Create output directory on scratch\nmkdir $TMPDIR/output_dir\n\n#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.\necho "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log\npython $HOME/thesis/run_layer_decomposition.py \\\n            --model_setup 4 \\\n            --memory_setup 1 \\\n            --img_dir $TMPDIR/video \\\n            --initial_mask {mask_argument} \\\n            --out_dir $TMPDIR/output_dir \\\n            --propagation_model $TMPDIR/weights/propagation_model.pth \\\n            --flow_model $TMPDIR/weights/flow_model.pth \\\n            --depth_model $TMPDIR/weights/depth_model.pth \\\n            --batch_size 4 \\\n            --n_epochs {config.n_epochs} \\\n            --save_freq 500 \\\n            --conv_channels 64 \\\n            --keydim 64 \\\n            --valdim 128 \\\n            --timesteps 4 \\\n            --description '{config.description}'"""
    
    arguments = ""
    for arg, value in config.arguments.items():
        arguments += " \\\n" + " "*12 + "--" + arg + " " + str(value)

    return header + body + arguments + footer

def generate_master_script(outdir, config):
    
    scripts = [f"{video}_{experiment}" for experiment in config.configs.keys() for video in config.videos.keys()]
    
    header = f"""#!/bin/bash\n\n"""
    forloop = f"""for EXPERIMENT in {" ".join(scripts)} \ndo\n    sbatch {outdir}/$EXPERIMENT.sh\ndone\n"""

    return header + forloop

def main(args):
    outdir = path.join(args.root, args.experiments.split(".")[0])
    makedirs(outdir, exist_ok=True)

    with open(f"{args.root}/{args.experiments}", "r") as f:
        data = EasyDict(json.load(f))

    videos = data.videos
    setups = data.configs

    for video, vid_data in videos.items():
        for name, exp in setups.items():
            experiment = EasyDict(exp)
            vid_data = EasyDict(vid_data)
            vid_data.name = video

            with open(f"{outdir}/{video}_{name}.sh", "w") as f:
                identifier = args.experiments.split(".")[0] + "__" + name
                f.write(generate_script(experiment, vid_data, identifier))

    with open(f"{outdir}/master.sh", "w") as f:
        f.write(generate_master_script(outdir, data))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root", type=str, default="job_scripts/experiments", help="root directory of the experiment scripts")
    parser.add_argument("--experiments", type=str, default="alpha_study.json", help="path to json file containing experiment configurations")

    args = parser.parse_args()

    main(args)