import torch
import torch.nn as nn
import numpy as np

from enum import Enum
import torchvision
from torch.nn import GroupNorm, Sequential

from tools.train_utils.torch_tools import change_default_args


class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args
    
class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"

rpn_class_dict = {
            "RPN": RPN,
        }

# rpn_class = rpn_class_dict[rpn_class_name]

# rpn = rpn_class(
#     use_norm=True,
#     num_class=num_class,
#     layer_nums=rpn_layer_nums,
#     layer_strides=rpn_layer_strides,
#     num_filters=rpn_num_filters,
#     upsample_strides=rpn_upsample_strides,
#     num_upsample_filters=rpn_num_upsample_filters,
#     num_input_filters=num_rpn_input_filters,
#     num_anchor_per_loc=target_assigner.num_anchors_per_location,
#     encode_background_as_zeros=encode_background_as_zeros,
#     use_direction_classifier=use_direction_classifier,
#     use_bev=use_bev,
#     use_groupnorm=use_groupnorm,
#     num_groups=num_groups,
#     box_code_size=target_assigner.box_coder.code_size)

# usage:
# preds_dict = self.rpn(spatial_features)