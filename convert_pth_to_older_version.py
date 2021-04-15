import torch
import argparse
from os import path, listdir

def main(args):
    checkpoints = [fn for fn in listdir(args.weight_dir) if fn.endswith(".pth")]
    for checkpoint_name in checkpoints:
        checkpoint = torch.load(path.join(args.weight_dir, checkpoint_name))
        torch.save(checkpoint, path.join(args.weight_dir, "zip_serialization_false", checkpoint_name), _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("weight_dir", type=str, help="Directory in which to look for checkpoints")
    
    args = parser.parse_args()

    main(args)