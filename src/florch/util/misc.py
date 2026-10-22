import numpy as np
import torch

from .easydict import EasyDict


def attrdict2dict_nested(elem):
    if isinstance(elem, dict):
        return {k: attrdict2dict_nested(v) for k, v in elem.items()}
    elif isinstance(elem, tuple):
        return tuple([attrdict2dict_nested(e) for e in elem])
    elif isinstance(elem, list):
        return [attrdict2dict_nested(e) for e in elem]
    else:
        return elem


def dict2attrdict_nested(elem):
    if isinstance(elem, dict):
        return EasyDict({k: dict2attrdict_nested(v) for k, v in elem.items()})
    elif isinstance(elem, tuple):
        return tuple([dict2attrdict_nested(e) for e in elem])
    elif isinstance(elem, list):
        return [dict2attrdict_nested(e) for e in elem]
    else:
        return elem


def smootherstep(x, x0=0, x1=None, length=None):
    """Ken Perlin's smootherstep: 6t^5 - 15t^4 + 10t^3."""
    if length is None:
        length = x1 - x0
    y = np.clip((x - x0) / length, 0, 1)
    return y * y * y * (y * (y * 6.0 - 15.0) + 10.0)


def new_rng(rng, advance_delta=0):
    """Derive a new independent RNG from an existing one."""
    gen = np.random.PCG64(rng.integers(0, 2**32 - 1))
    gen.advance(advance_delta)
    return np.random.Generator(gen)


def to_device_nested(elem, device):
    if isinstance(elem, dict):
        return {k: to_device_nested(v, device) for k, v in elem.items()}
    elif isinstance(elem, tuple):
        return tuple([to_device_nested(e, device) for e in elem])
    elif isinstance(elem, list):
        return [to_device_nested(e, device) for e in elem]
    elif isinstance(elem, torch.Tensor):
        return elem.to(device, non_blocking=True)
    else:
        return elem
