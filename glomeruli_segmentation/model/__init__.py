import functools

import torch

from .unet import UNet


def create(
    arch,
    backbone,
    activation=None,
    pretrained="imagenet",
    frozen_layers=0,
    n_classes=8,
    feature_layers=None,
    frozen_batchnorm=True,
    objectness=False,
    tta=0,
    scales=None,
    resize=None,
):
    if arch.startswith("unet"):
        arch_split = arch.split("_")
        decoder = arch_split[1] if len(arch_split) > 1 else "simple"
        image_classification = arch_split[2] == "c" if len(arch_split) > 2 else False
        model = UNet()
    else:
        raise NotImplementedError(arch)
    model.load = load_model
    return model


def load(filename, create, **kwargs):
    print("load {}".format(filename))
    data = torch.load(filename, map_location="cpu")
    data["kwargs"].update(kwargs)

    net = create(*data["args"], **data["kwargs"])
    net.load_state_dict(data["state_dict"])
    return net


load_model = functools.partial(load, create=create, pretrained=False)
