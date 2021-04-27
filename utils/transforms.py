import random

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            output = t(inputs)
        return output


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, inputs):
        output = []
        if random.random() < self.prob:
            for input in inputs:
                output.append(input.flip(-1))
        return output


class ToTensor(object):
    def __call__(self, inputs):
        output = []
        for input in inputs:
            output.append(F.to_tensor(input))
        return output

    
def get_transforms(train):
    transforms = []
    transforms.append(ToTensor())
    return Compose(transforms)
