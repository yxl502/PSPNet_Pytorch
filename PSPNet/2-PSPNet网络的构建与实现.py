import torch
import torch.nn as nn
import torch.nn.functional as F


class conv2dBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, bias):
        super(conv2dBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding,
                              dilation, bias=bias)

        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)
        return outputs


class FeatureMap_convolution(nn.Module):
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        in_channels, out_channels, kernel_size, stride, padding, dialation, bias = \
            3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dialation, bias
        )

        in_channels, out_channels, kernel_size, stride, padding, dialation, bias = \
            64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dialation, bias
        )

        in_channels, out_channels, kernel_size, stride, padding, dialation, bias = \
            64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size, stride, padding, dialation, bias
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs


class conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, bias):
        super(conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding,
                              dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class bottleNeckPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride,
                 dilation):
        super(bottleNeckPSP, self).__init__()
        self.cbr_1 = conv2dBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )
        self.cbr_2 = conv2dBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.cbr_3 = conv2dBatchNorm(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )

        self.cb_residual = conv2dBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride,
            padding=0, dilation=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = conv2dBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )
        self.cbr_2 = conv2dBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
            dilation=dilation, bias=False
        )
        self.cbr_3 = conv2dBatchNorm(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cbr_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels,
                 stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        self.add_module(
            'block1',
            bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )

        for i in range(n_blocks - 1):
            self.add_module(
                'block' + str(i+2),
                bottleNeckIdentifyPSP(
                    out_channels, mid_channels, stride, dilation
                )
            )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()
        self.height = height
        self.width = width

        out_channels = int(in_channels / len(pool_sizes))
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2dBatchNormRelu(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0,
            dilation=1, bias=False
        )

    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(
            self.height, self.width
        ), mode='bilinear', align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(
            self.height, self.width
        ), mode='bilinear', align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(
            self.height, self.width
        ), mode='bilinear', align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(
            self.height, self.width
        ), mode='bilinear', align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)
        return output


class DecodePSPFeature(nn.Module):
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width

        self.cbr = conv2dBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1,
            padding=1, dilation=1, bias=False
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=n_classes, kernel_size=1,
            stride=1, padding=0
        )

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width),
                               mode='bilinear', align_corners=True)
        return output


class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPlayers, self).__init__()

        self.height = height
        self.width = width
        self.cbr = conv2dBatchNormRelu(
            in_channels=in_channels, out_channels=256, kernel_size=3,
            stride=1, padding=1, dilation=1, bias=False
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256,
                                        out_channels=n_classes,
                                        kernel_size=1, stride=1,
                                        padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width),
                               mode='bilinear', align_corners=True)
        return output


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64,
            out_channels=256, stride=1, dilation=1
        )
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128,
            out_channels=512, stride=2, dilation=1
        )
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256,
            out_channels=1024, stride=1, dilation=2
        )
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512,
            out_channels=2048, stride=1, dilation=4
        )

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[
            6, 3, 2, 1
        ], height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(
            height=img_size, width=img_size, n_classes=n_classes
        )

        self.aux = AuxiliaryPSPlayers(
            in_channels=1024, height=img_size,
            width=img_size, n_classes=n_classes
        )

    def forward(self, x):
        # print('x0: ', x.size())     # x0:  torch.Size([2, 3, 475, 475])
        x = self.feature_conv(x)
        # print('x1: ', x.size())      # x1:  torch.Size([2, 128, 119, 119])
        x = self.feature_res_1(x)
        # print('x2: ', x.size())     # x2:  torch.Size([2, 256, 119, 119])
        x = self.feature_res_2(x)
        # print('x3: ', x.size())     # x3:  torch.Size([2, 512, 60, 60])
        x = self.feature_dilated_res_1(x)
        # print('x4: ', x.size())     # x4:  torch.Size([2, 1024, 60, 60])

        output_aux = self.aux(x)
        # print('output_aux: ', output_aux.size())    # output_aux:  torch.Size([2, 21, 475, 475])

        x = self.feature_dilated_res_2(x)
        # print('x5: ', x.size())     # x5:  torch.Size([2, 2048, 60, 60])

        x = self.pyramid_pooling(x)
        # print('x6: ', x.size())      # x6:  torch.Size([2, 4096, 60, 60])
        output = self.decode_feature(x)
        # print('output: ', output.size())    # output:  torch.Size([2, 21, 475, 475])

        return output, output_aux


net = PSPNet(n_classes=21)

print(net)

batch_size = 2
dummy_img = torch.rand(batch_size, 3, 475, 475)

outputs = net(dummy_img)

print(outputs[0].size())   # torch.Size([2, 21, 475, 475])
print(outputs[1].size())   # torch.Size([2, 21, 475, 475])


