from florch.util.easydict import EasyDict
import torch


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
