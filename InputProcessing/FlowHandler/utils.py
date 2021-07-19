class RaftNameSpace(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def _get_kwargs(self):
        return self.__dict__.keys()
