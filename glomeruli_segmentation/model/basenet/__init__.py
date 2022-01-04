import torch.nn as nn
import torchvision

from .basic import Sequential, get_num_of_channels


def create_basenet(
    name,
    pretrained,
    **kwargs,
):
    if name == "Resnet34":
        pool_in_2nd = name[0].isupper()
        imagenet_pretrained = pretrained == "imagenet"

        model = torchvision.models.resnet34(pretrained=imagenet_pretrained, **kwargs)
        maxpool = model.maxpool if hasattr(model, "maxpool") else model.maxpool1
        if pool_in_2nd:
            layer0 = Sequential(model.conv1, model.bn1, model.relu)
        else:
            layer0 = Sequential(model.conv1, model.bn1, model.relu, maxpool)

        layer0[-1].out_channels = get_num_of_channels(model.conv1)

        def get_out_channels_from_resnet_block(layer):
            block = layer[-1]
            block_name = block.__class__.__name__
            if "BasicBlock" in block_name or "SEResNetBlock" in block_name:
                return block.conv2.out_channels
            elif "Bottleneck" in block_name:
                return block.conv3.out_channels
            raise RuntimeError("unknown resnet block: {}".format(block.__class__))

        layer1 = model.layer1
        layer2 = model.layer2
        layer3 = model.layer3
        layer4 = model.layer4

        layer1.out_channels = layer1[-1].out_channels = get_out_channels_from_resnet_block(model.layer1)
        layer2.out_channels = layer2[-1].out_channels = get_out_channels_from_resnet_block(model.layer2)
        layer3.out_channels = layer3[-1].out_channels = get_out_channels_from_resnet_block(model.layer3)
        layer4.out_channels = layer4[-1].out_channels = get_out_channels_from_resnet_block(model.layer4)

        if pool_in_2nd:
            layer1 = nn.Sequential(maxpool, layer1)
            layer1.out_channels = layer1[-1].out_channels

        n_pretrained = 5 if imagenet_pretrained else 0

        return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained
    else:
        raise NotImplementedError(name)
