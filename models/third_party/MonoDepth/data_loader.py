from os import path, listdir
from PIL import Image

from torch.utils.data import Dataset


class FrameLoader(Dataset):
    def __init__(self, root_dir, transform=None):

        self.frame_paths = [path.join(root_dir, fname) for fname in sorted(listdir(root_dir))]
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        frame = Image.open(self.frame_paths[idx])
        if self.transform:
            frame = self.transform(frame)
        return frame
