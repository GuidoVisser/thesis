

config = {
    "directories": {
        'out_dir':           'results/layer_decomposition_dynamic/tennis', 
        'initial_mask':      'datasets/DAVIS/Annotations/480p/tennis/00000.png', 
        'img_dir':           'datasets/DAVIS/JPEGImages/480p/tennis',
        'propagation_model': 'models/third_party/weights/propagation_model.pth', 
        'flow_model':        'models/third_party/weights/raft-things.pth'
    },
    "reconstruction_model": {
        'coarseness':      10, 
        'device':          'cuda:0', 
        'composite_order': None 
    },
    "training_parameters": {
        'batch_size':           1, 
        'learning_rate':        0.001,
        'memory_learning_rate': 0.001,
        'n_epochs':             300, 
        'save_freq':            30, 
        'n_gpus':               1, 
        'seed':                 1 
    },
    "memory_network" : {
        'keydim':     128, 
        'valdim':     512, 
        'mem_freq':   30, 
        'mem_device': 'cuda:0'
    }
}

def update_config(args):
    """
    Update the config with the current settings

    Args:
        args (namespace): namespace containing settings of the model
    """

    config["x"] = 2


def save_config(filepath, config):
    """
    Save the config to a .txt file so the experiment settings can be reveiwed later
    """
    def recursive_write(input, io_stream, depth):

        for item in input.items():
            if isinstance(input[item], dict):
                recursive_write(input[item], io_stream, depth + 1)
            else:
                io_stream.write(f"{'\t'*depth}{item} -- {input[item]}\n")


    with open(filepath, "a") as txt_file:
        recursive_write(config, txt_file, 0)


