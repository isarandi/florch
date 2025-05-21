class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and not k in ("update", "pop"):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        super().__setitem__(name, value)

    def __getattr__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError:
            raise AttributeError(name)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k, v in d.items():
            setattr(self, k, v)

    def pop(self, k, d=None):
        delattr(self, k)
        return super().pop(k, d)
