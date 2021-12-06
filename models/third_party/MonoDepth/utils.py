from torch.utils.data import DataLoader

from .data_loader import FrameLoader
from .transforms import image_transforms

def prepare_dataloader(data_directory, size, num_workers=4):
    data_transform = image_transforms(size = size)

    dataset = FrameLoader(data_directory, transform=data_transform)
    n_img = len(dataset)
    loader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    return n_img, loader
