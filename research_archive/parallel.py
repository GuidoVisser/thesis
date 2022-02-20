from torch import nn

class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, name: str):
        return getattr(self.module, name)


