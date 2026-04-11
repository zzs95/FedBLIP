""" Code adapted from https://github.com/hassony2/inflated_convnets_pytorch """

import math

from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models import inflate
# import inflate
import torchvision
import copy

class I3ResNet(nn.Module):
    def __init__(
        self, resnet2d, frame_nb=16, class_nb=1000, conv_class=False, return_skips=False, return_pool=False
    ):
        """
        Args:
            conv_class: Whether to use convolutional layer as classifier to
                adapt to various number of frames
        """
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.return_pool = return_pool
        self.conv_class = conv_class

        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=True
        )
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2
        )

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate.inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True,
            )
            self.contrastive_head = nn.Conv3d(
                in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate.inflate_pool(
                resnet2d.avgpool, time_dim=final_time_dim
            )
            # self.fc = inflate.inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        if self.conv_class:
            x_features = self.avgpool(x)
            x_ehr = self.classifier(x_features)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            x_contrastive = self.contrastive_head(x_features)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)

            if self.return_skips and self.return_pool:
                return x_contrastive, x_ehr, skips, x_features
            elif self.return_skips and not self.return_pool:
                return x_contrastive, x_ehr, skips
            elif not self.return_skips and self.return_pool:
                return x_contrastive, x_ehr, x_features
            else:
                return x_contrastive, x_ehr
        else:
            x_features = self.avgpool(x)
            if self.return_skips and self.return_pool:
                return skips, x_features
            else:
                x_features



def inflate_reslayer(reslayer2d):
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return nn.Sequential(*reslayers3d)


class Bottleneck3d(nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out
        
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if x.requires_grad:
            out = checkpoint.checkpoint(run_function, x)
        else:
            out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=True
        ),
        inflate.inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d

    
if __name__ == '__main__':
    # from torchsummary import summary
    # model = ImageClassifier(1).cuda()
    import sys
    sys.path.append('/media/brownradx/ssd_code/Projects_zhusi/PE_data_process/pe_25_code/')
    from abnormality_list import abnormality_dict
    region_classes = []
    for k in abnormality_dict.keys():
        region_classes.append(len(abnormality_dict[k]))
    from models.image_classifier import ImageClassifier_RegionHeads
    model = ImageClassifier_RegionHeads(region_classes).cuda()
    # Calculate total number of trainable parameters
    # summary(model.to('cuda'), (1, 128, 128, 128))
    image_pt = torch.rand((2, 1, 224, 224, 160)) .cuda()
    out_cls = model(image_pt)
    print(out_cls[1].shape)

