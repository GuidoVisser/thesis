import random

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs):
        for t in self.transforms:
            inputs = t(inputs)
        return inputs

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

class ImagePadder(object):
    def __call__(self, inputs):
        output = []
        for input in inputs:
            output.append(F.pad(input, (1,0)))
        return output

class Normalize(object):
    def __call__(self, inputs):
        output = []
        for input in inputs:
            if input.size()[0] == 3:
                output.append(F.normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            else:
                output.append(input)
        return output

class DeNormalize(object):
    def __call__(self, inputs):
        output = []
        for input in inputs:
            if input.size()[0] == 3:
                output.append(F.normalize(input, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]))
            else:
                output.append(input)
        return output
    
def get_transforms():
    transforms = []
    transforms.append(ToTensor())
    # transforms.append(ImagePadder())
    transforms.append(Normalize())
    return Compose(transforms)
