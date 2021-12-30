import json
from easydict import EasyDict
from argparse import ArgumentParser
from os import makedirs, path

def generate_script(config, identifier):
    header = f"""#!/bin/bash \n\n#SBATCH -n 1\n#SBATCH -t {config.time_estimate}\n#SBATCH -p gpu\n#SBATCH --gpus-per-node=gtx1080ti:4\n\n# get start time of script\nDT=`date +"%m_%d_%H_%M_%S"`\n\n# load modules\nmodule load 2020\nmodule load Python/3.8.2-GCCcore-9.3.0\n\n# install dependencies\npip install --user --upgrade torch && pip install --user --upgrade torchvision\n\n#Copy input file to scratch\nVIDEO='{config.video}'\nDATASET='{config.dataset}'\nMASK_PATH='{config.mask_path}'\ncp -RT $HOME/thesis/datasets/$DATASET/JPEGImages/480p/$VIDEO $TMPDIR/video\ncp -RT $HOME/thesis/datasets/$DATASET/Annotations/$VIDEO/combined/$MASK_PATH $TMPDIR/$MASK_PATH\nmkdir $TMPDIR/weights\ncp $HOME/thesis/models/third_party/weights/topkstm.pth $TMPDIR/weights/propagation_model.pth\ncp $HOME/thesis/models/third_party/weights/raft.pth $TMPDIR/weights/flow_model.pth\ncp $HOME/thesis/models/third_party/weights/monodepth.pth $TMPDIR/weights/depth_model.pth\n\n#Create output directory on scratch\nmkdir $TMPDIR/output_dir\n\n#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.\necho "$SLURM_JOBID | Start: $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log\npython $HOME/thesis/run_layer_decomposition.py \\\n            --model_setup 4 \\\n            --memory_setup 1 \\\n            --img_dir $TMPDIR/video \\\n            --initial_mask $TMPDIR/$MASK_PATH \\\n            --out_dir $TMPDIR/output_dir \\\n            --propagation_model $TMPDIR/weights/propagation_model.pth \\\n            --flow_model $TMPDIR/weights/flow_model.pth \\\n            --depth_model $TMPDIR/weights/depth_model.pth \\\n            --batch_size 4 \\\n            --n_epochs 250 \\\n            --save_freq 50 \\\n            --conv_channels 64 \\\n            --keydim 64 \\\n            --valdim 128 \\\n            --timesteps 4 \\\n            --description '{config.description}'"""
    footer = f"""echo "$SLURM_JOBID | End:   $(date)" >> $HOME/thesis/job_logs/run_layer_decomposition.log\n\n#Copy output directory from scratch to home\nmkdir -p $HOME/thesis/results/layer_decomposition/$VIDEO__$SLURM_JOBID__{identifier}\ncp -RT $TMPDIR/output_dir $HOME/thesis/results/layer_decomposition/$VIDEO__$SLURM_JOBID__{identifier}\n"""
    arguments = ""
    for arg, value in config.arguments.items():
        arguments += " \\ \n" + " "*12 + "--" + arg + " " + value

    return header + arguments + footer

def generate_master_script(outdir, config):
    header = f"""#!/bin/bash\n\n"""

    forloop = f"""for EXPERIMENT in {" ".join(config.keys())} \ndo\n    sbatch {outdir}/$EXPERIMENT.sh\ndone\n"""

    return header + forloop

def main(args):
    outdir = path.join(args.root, args.experiments.split(".")[0])
    makedirs(outdir, exist_ok=True)

    with open(f"{args.root}/{args.experiments}", "r") as f:
        setups = EasyDict(json.load(f))

    for name, exp in setups.items():
        experiment = EasyDict(exp)

        with open(f"{outdir}/{name}.sh", "w") as f:
            identifier = args.experiments.split(".")[0] + "__" + name
            f.write(generate_script(experiment, identifier))

    with open(f"{outdir}/master.sh", "w") as f:
        f.write(generate_master_script(outdir, setups))

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root", type=str, default="job_scripts/experiments", help="root directory of the experiment scripts")
    parser.add_argument("--experiments", type=str, default="setups.json", help="path to json file containing experiment configurations")

    args = parser.parse_args()

    main(args)