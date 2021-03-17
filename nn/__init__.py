import torch
import functools

from .unet import UNet

def create(arch, backbone, activation=None, pretrained='imagenet', frozen_layers=0, n_classes=8,
           feature_layers=None, frozen_batchnorm=True, objectness=False, tta=0, scales=None, resize=None):
    if arch.startswith('unet'):
        arch_split = arch.split('_')
        decoder = arch_split[1] if len(arch_split) > 1 else 'simple'
        image_classification = arch_split[2] == 'c' if len(arch_split) > 2 else False
        model = UNet(backbone, pretrained=pretrained, n_classes=n_classes,
                     activation=activation, frozen_layers=frozen_layers,
                     objectness=objectness, tta=tta, scales=scales, resize=resize,
                     num_head_features=16 * n_classes, cat_features=True, ocnet=False, decoder=decoder,
                     image_classification=image_classification, frozen_batchnorm=frozen_batchnorm
                     )
    else:
        raise NotImplementedError(arch)
    model.load = load_model
    return model


def load(filename, create, **kwargs):
    print('load {}'.format(filename))
    data = torch.load(filename, map_location='cpu')
    data['kwargs'].update(kwargs)

    net = create(*data['args'], **data['kwargs'])
    net.load_state_dict(data['state_dict'])
    return net


load_model = functools.partial(load, create=create, pretrained=False)
