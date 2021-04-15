from os import path, makedirs

def create_dir(dir_path):
    """
    Creates a directory if not exist.
    """
    if not path.exists(dir_path):
        makedirs(dir_path)

def collate_fn(batch):
    return tuple(zip(*batch))
