import functools

import torch
import torch.nn.functional as F
from torch import nn

from .basenet import create_basenet
from .basenet.basic import ConvBn, ConvBnRelu, ConvRelu
from .basenet.scse import SCSEBlock
from .segnet import SegNet


class UpsamplingBilinear(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        if self.scale_factor == 1:
            return input
        return F.interpolate(input, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class DecoderBase(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, scale_factor=2):
        super().__init__()
        self.out_channels = out_channels
        self.block = self._init(in_channels, middle_channels, out_channels, scale_factor)

    def _init(self, in_channels, middle_channels, out_channels, scale_factor):
        raise NotImplementedError

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


class DecoderSimple(DecoderBase):
    """as dsb2018_topcoders
    from https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/models/unets.py#L68
    """

    def _init(self, in_channels, middle_channels, out_channels, scale_factor):
        return nn.Sequential(
            ConvBnRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            UpsamplingBilinear(scale_factor),
            ConvBn(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class DecoderSimpleNBN(DecoderBase):
    """as dsb2018_topcoders
    from https://github.com/selimsef/dsb2018_topcoders/blob/master/selim/models/unets.py#L76
    """

    def _init(self, in_channels, middle_channels, out_channels, scale_factor):
        return nn.Sequential(
            ConvRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            UpsamplingBilinear(scale_factor),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class DecoderDeConv(DecoderBase):
    def _init(self, in_channels, middle_channels, out_channels, scale_factor):
        assert scale_factor == 2
        return nn.Sequential(
            # nn.Dropout2d(p=0.1, inplace=True),
            ConvBn(in_channels, middle_channels, kernel_size=3, padding=1),
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # upsample(scale_factor=2)
        )


class DecoderSCSE(DecoderBase):
    """
    https://github.com/SeuTao/TGS-Salt-Identification-Challenge-2018-_4th_place_solution/blob/master/model/model.py#L125
    """

    def _init(self, in_channels, middle_channels, out_channels, scale_factor):
        return nn.Sequential(
            # SCSEBlock(in_channels),
            ConvBnRelu(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            ConvBnRelu(middle_channels, out_channels, kernel_size=3, padding=1, bias=False),
            SCSEBlock(out_channels),
            UpsamplingBilinear(scale_factor),
        )


class UNet(SegNet):
    MAX_PREDICT_WINDOW = 1024

    def __init__(
        self,
        backbone="Resnet50",
        num_filters=16,
        n_classes=1,
        pretrained="imagenet",
        activation=None,
        frozen_layers=0,
        objectness=False,
        tta=0,
        scales=None,
        resize=None,
        ocnet=False,
        align_corners=False,
        decoder="simple",
        dropout=0.1,
        num_head_features=None,
        cat_features=False,
        deep_supervision=False,
        frozen_batchnorm=False,
        refine=False,
        image_classification=False,
    ):
        super().__init__(objectness=objectness, tta=tta, scales=scales, resize=resize)
        Decoder = dict(
            simple=DecoderSimple,
            noBN=DecoderSimpleNBN,
            deconv=DecoderDeConv,
            scse=DecoderSCSE,
        )[decoder]
        self.align_corners = align_corners

        net, _, _ = create_basenet(
            backbone,
            activation=activation,
            pretrained=pretrained,
            frozen_batchnorm=frozen_batchnorm,
            frozen_layers=frozen_layers,
            # replace_stride_with_dilation=[False, True, True]
        )

        self.encoder1 = nn.Sequential(net[0], nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 64  64
        self.encoder2 = nn.Sequential(net[1][1])  # 64  256
        self.encoder3 = net[2]  # 128 512
        self.encoder4 = net[3]  # 256 1024
        self.encoder5 = net[4]  # 512 2048

        context_channels = num_filters * 8 * 4
        self.center = nn.Sequential(
            nn.Conv2d(self.encoder5.out_channels, context_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(context_channels),
            nn.ReLU(),
        )

        self.decoder5 = Decoder(
            self.encoder5.out_channels + context_channels,
            num_filters * 16,
            num_filters * 16,
        )
        self.decoder4 = Decoder(
            self.encoder4.out_channels + self.decoder5.out_channels,
            num_filters * 8,
            num_filters * 8,
        )
        self.decoder3 = Decoder(
            self.encoder3.out_channels + self.decoder4.out_channels,
            num_filters * 4,
            num_filters * 4,
        )
        self.decoder2 = Decoder(
            net[1].out_channels + self.decoder3.out_channels,
            num_filters * 2,
            num_filters * 2,
            scale_factor=1,
        )
        self.decoder1 = Decoder(
            net[0].out_channels + self.decoder2.out_channels,
            num_filters,
            num_filters,
            scale_factor=1,
        )

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.num_classes = n_classes

        self.cat_features = cat_features
        feature_channels = (
            self.decoder1.out_channels,
            self.decoder2.out_channels,
            self.decoder3.out_channels,
            self.decoder4.out_channels,
            self.decoder5.out_channels,
        )
        neck_in_channels = sum(feature_channels) if cat_features else self.decoder1.out_channels
        self.neck = nn.Sequential(
            nn.Conv2d(
                neck_in_channels,
                num_head_features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_head_features),
            nn.ReLU(),
        )
        self.mask_head = nn.Conv2d(num_head_features, n_classes, kernel_size=3, padding=1)

        self.deep_supervision_heads = None
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList(
                [
                    nn.Conv2d(d.out_channels, n_classes, kernel_size=3, padding=1)
                    for d in [
                        self.decoder5,
                        self.decoder4,
                        self.decoder3,
                        self.decoder2,
                        self.decoder1,
                    ]
                ]
            )

    def _forward(self, x):
        e1 = self.encoder1(x)  # ; print('e1', e1.size())
        e2 = self.encoder2(e1)  # ; print('e2', e2.size())
        e3 = self.encoder3(e2)  # ; print('e3', e3.size())
        e4 = self.encoder4(e3)  # ; print('e4', e4.size())
        e5 = self.encoder5(e4)  # ; print('e5', e5.size())
        # c = self.center(self.pool(e5))#; print('c', c.size())
        c = self.center(e5)  # ; print('c', c.size())

        d5 = self.decoder5(c, e5)  # ; print('d5', d5.size())
        d4 = self.decoder4(d5, e4)  # ; print('d4', d4.size())
        d3 = self.decoder3(d4, e3)  # ; print('d3', d3.size())
        d2 = self.decoder2(d3, e2)  # ; print('d2', d2.size())
        d1 = self.decoder1(d2, e1)  # ; print('d1', d1.size())

        if self.cat_features:
            d1_size = d1.size()[2:]
            upsampler = functools.partial(
                F.interpolate,
                size=d1_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            us = [upsampler(d) for d in (d5, d4, d3, d2)] + [d1]
            # ds = [self.dropout(u) for u in us]
            # d = torch.cat(ds, 1)
            d = torch.cat(us, 1)
            d = self.dropout(d)
        else:
            d = self.dropout(d1)

        d = self.neck(d)
        mask = self.mask_head(d)  # ; print("mask", mask.size())

        if self.training:
            outputs = dict(out=mask)
            if self.deep_supervision_heads:
                features = (d5, d4, d3, d2, d1)
                for i, (m, f) in enumerate(zip(self.deep_supervision_heads, features)):
                    outputs["aux" + str(i)] = m(f)

            if len(outputs) > 1:
                return outputs

        return mask


if __name__ == "__main__":
    net = UNet(backbone="Resnet50", pretrained=False)
    img = torch.zeros((3, 256, 256))
    # target = torch.zeros((1, net.image_size[0], net.image_size[1]), dtype=torch.long)
    outputs = net.predict(img)
    print(outputs.shape)
    # loss = net.criterion(outputs, {'mask': target}, loss_func=torch.nn.functional.binary_cross_entropy_with_logits)
    # print(loss)
