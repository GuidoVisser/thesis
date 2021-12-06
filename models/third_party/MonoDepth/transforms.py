import torch
import torchvision.transforms as transforms

def image_transforms(size=(256, 512)):
    data_transform = transforms.Compose([
        ResizeImage(size=size),
        ToTensor(),
        DoTest(),
    ])
    return data_transform


class ResizeImage(object):
    def __init__(self, size=(256, 512)):
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        return self.transform(sample)


class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        return self.transform(sample)


class DoTest(object):
    def __call__(self, sample):
        return torch.stack((sample, torch.flip(sample, [2])))
