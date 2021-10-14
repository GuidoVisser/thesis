

config = {
    "x": 1
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


