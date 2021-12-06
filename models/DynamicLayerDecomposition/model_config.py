from argparse import Namespace
import os
import re

def default_config():
    return {
        "description": "no description given ",
        "directories": {
            'out_dir':           'results/layer_decomposition_dynamic/tennis', 
            'initial_mask':      'datasets/DAVIS/Annotations/480p/tennis/00000.png', 
            'img_dir':           'datasets/DAVIS/JPEGImages/480p/tennis',
            'propagation_model': 'models/third_party/weights/propagation_model.pth', 
            'flow_model':        'models/third_party/weights/raft-things.pth'
        },
        "model": {
            'model_type':                '',
            'shared_backbone':           False,
            'coarseness':                10, 
            'in_channels':               16,
            'memory_in_channels':        16,
            'conv_channels':             64,
            'keydim':                    128, 
            'valdim':                    512,
            'use_2d_loss_module':        False,
            'no_static_background':      False,
            'memory_t_strided':          False
        },
        "model input": {
            'memory_input_type':         '',
            'num_static_channels':       5,
            'noise_temporal_coarseness': 2,
            'timesteps':                 16,
            'memory_timesteps':          16,
            'mem_freq':                  1,
            'frame_height':              256,
            'frame_width':               448,
            'jitter_rate':               0.75,
            'composite_order':           None, 
        },
        "training_parameters": {
            'batch_size':                1, 
            'learning_rate':             0.001,
            'device':                    'cuda:0',
            'n_epochs':                  300, 
            'save_freq':                 30, 
            'n_gpus':                    1, 
            'seed':                      1, 
            'alpha_bootstr_rolloff':     50,
            'alpha_loss_l1_rolloff':     100, 
        },
        "lambdas": {
            "lambda_mask":               50.,
            "lambda_recon_flow":         1.,
            "lambda_recon_warp":         0.,
            "lambda_recon_depth":        0.5,
            "lambda_alpha_warp":         0.005,
            "lambda_alpha_l0":           0.005,
            "lambda_alpha_l1":           0.01,
            "lambda_stabilization":      0.001,
            "lambda_dynamics_reg_diff":  0.0005,
            "lambda_dynamics_reg_corr":  0.001,
        },
        "pretrained_models" : {
            "propagation_model": "models/third_party/weights/propagation_model.pth",
            "flow_model": "models/third_party/weights/raft-things.pth",
            "depth_model": "models/third_party/weights/monodepth_resnet18_001.pth"
        }
    }

def update_config(args: Namespace, config: dict) -> dict:
    """
    Update the config with the current settings

    Args:
        args (namespace): namespace containing settings of the model
        config (dict): dictionary containing the new config settings

    Returns:
        config (dict): updated configuration
    """
    def recursive_update(arg_dict, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                recursive_update(arg_dict, value)
            elif key in arg_dict:
                dictionary[key] = arg_dict[key]
            else:
                continue

    recursive_update(vars(args), config)

    return config


def save_config(filepath: str, config: dict, mode: str = "a") -> None:
    """
    Save the config to a .txt file so the experiment settings can be reviewed later

    Args:
        filepath (str): path to save location
        config (dict): configuration to be saved
        mode (str): mode of saving, either 'append' or 'read'
    """
    assert mode in ["a", "w"], f"incorrect mode given. Expected either 'a' or 'w' but got '{mode}'."

    def recursive_write(input, io_stream, depth):

        for (item, value) in input.items():
            if item == "description":
                value = re.sub("(.{64})", "\\1\n\t", value, 0, re.DOTALL)
                io_stream.write(f"{item} -- \n\t{value}\n")
            elif isinstance(input[item], dict):
                io_stream.write(f"{''.join([' ']*4*depth)}{item}:\n")
                recursive_write(value, io_stream, depth + 1)
            else:
                first_part = f"{''.join([' ']*4*(depth))}{item}".ljust(25)
                io_stream.write(f"{first_part} -- {value}\n")


    with open(filepath, mode) as txt_file:
        recursive_write(config, txt_file, 0)

def load_config(filepath: str) -> dict:
    """
    Load the config from a .txt file
    
    Args:
        filepath (str): path to config file
    
    Returns:
        config (dict): new configuration
    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    default = default_config()

    config = {}
    header = ""
    is_description = False
    for line in lines:
        if line.startswith("\t") or line.startswith(" "):
            if is_description:
                config["description"] += line.strip("\n \t") + " "
            else:
                key, value = line.split("--")
                key = key.strip("\t ")
                value = value.strip("\n ")

                if value == "None":
                    value = None
                elif type(default[header][key]) == bool:
                    value = value == "True"
                elif value.startswith("["):
                    value = [entry.strip("[] \'") for entry in value.split(",")]
                else:
                    value = type(default[header][key])(value)
                config[header][key] = value
        elif line.startswith("description"):
            is_description = True
            config["description"] = ""
        else:
            header = line.strip(": \n")
            config[header] = {}
            is_description = False

    return config

def read_config(args: Namespace, config: dict) -> Namespace:
    """
    Override the namespace arguments with the values in the given config dictionary

    Args:
        args (Namespace)
        config (dict)

    returns:
        args (Namespace)
    """

    def recursive_update(arg_dict, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                recursive_update(arg_dict, value)
            elif key in arg_dict:
                arg_dict[key] = dictionary[key]
            else:
                continue

    # create a dictionary liaison version of the NameSpace args 
    # Updating the liaison will immediately update the corresponding values in args
    liaison = vars(args)
    recursive_update(liaison, config)

    return args


if __name__ == "__main__":
    save_config(os.getcwd() + "/test.txt", default_config())

    config = load_config(os.getcwd() + "/test.txt")
