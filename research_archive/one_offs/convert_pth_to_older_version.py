import torch
import argparse
from os import path, listdir

def convert_pth_to_older_version(weight_dir):
    """
    PyTorch version <1.6.0 use a different serialization method for model checkpoints. 
    Convert newer version to older verion

    NOTE
    ! make sure you run this script in an environment with PyTorch version >=1.6 !

    Args:
        weight_dir (str): path to directory. All .pth files in that directory will be copied and converted
    """
    # check pytorch version
    assert torch.__version__.split("+")[0] >= "1.6.0", "Please run this script with PyTorch version >= 1.6.0"

    # copy and convert all .pth files in weight_dir to have zip_serialization = False
    checkpoints = [fn for fn in listdir(weight_dir) if fn.endswith(".pth")]
    for checkpoint_name in checkpoints:
        checkpoint = torch.load(path.join(weight_dir, checkpoint_name))
        torch.save(checkpoint, path.join(weight_dir, "zip_serialization_false", checkpoint_name), _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("weight_dir", type=str, help="Directory in which to look for checkpoints")
    
    args = parser.parse_args()

    convert_pth_to_older_version(args.weight_dir)