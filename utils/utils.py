from os import path, makedirs

def create_dir(dir_path):
    """
    Creates a directory if not exist.
    """
    makedirs(dir_path, exist_ok=True)

def collate_fn(batch):
    return tuple(zip(*batch))
