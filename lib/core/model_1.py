import torch 
import torch.nn as nn

from typing import List, Sequence, Optional
from einops import rearrange
from functools import partial

import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
class Spatial_SE(nn.Module):
    def __init__(
        self,
        in_channels,
        res):
        super(Spatial_SE, self).__init__()
        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1), #, groups = in_channels),
            nn.Hardswish(inplace = True),
            nn.Conv2d(in_channels // 16, 1, 1),
            nn.Conv2d(1, 1, res, stride = res),
            nn.Hardsigmoid(),
            nn.UpsamplingNearest2d(scale_factor = res)
        )

    def forward(self, x):
        scale = self.spatial_se(x)
        x = x * scale
        return x

class CA(nn.Module):
    def __init__(self, inp, reduction = 8):
        super(CA, self).__init__()
        # h:height(行)   w:width(列)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (b,c,h,w)-->(b,c,h,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (b,c,h,w)-->(b,c,1,w)
 
 
         # mip = max(8, inp // reduction)  论文作者所用
        mip =  inp // reduction  # 博主所用   reduction = int(math.sqrt(inp))
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace = True)
 
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (b,c,w,1)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out

        
class Stem(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        groups = 1,
        dilation = 1,      
        norm_layer = nn.BatchNorm2d,
        activation = 'RE',
    ):
        super(Stem, self).__init__()
        self.activation = activation
        padding = (kernel_size - 1)//2 * dilation
    
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride = stride, padding = padding, dilation = dilation, 
            groups = groups, bias = False)

        self.norm_layer = norm_layer(out_channels, eps=0.01, momentum=0.01)
        if activation == 'PRE':
            self.acti_layer = nn.PReLU()
        elif activation == 'HS':
            self.acti_layer = nn.Hardswish(inplace = True)  
        else:
            self.acti_layer = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        if self.activation is not None:
            x = self.acti_layer(x)
        return x


class BlockConfig:
    def __init__(
        self,
        in_channels,
        exp_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        activation,
        use_se,
        use_softmax = False,
        res = 1,
        
        
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.use_se = use_se
        self.use_softmax = use_softmax
        self.res = res
        
class Block(nn.Module):
    def __init__(
        self,
        size,
        cnf: BlockConfig,
        ):
        super(Block, self).__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.in_channels == cnf.out_channels

        layers = []
        # expand
        if cnf.exp_channels != cnf.in_channels:
            layers.append(
                Stem(
                    cnf.in_channels, 
                    cnf.exp_channels, 
                    kernel_size=1,
                    activation=cnf.activation
                )
            )

        # depthwise
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.exp_channels,
                kernel_size=cnf.kernel_size,
                stride=cnf.stride,
                groups=cnf.exp_channels,
                dilation=cnf.dilation,
                activation=cnf.activation,
            )
        )

        if cnf.use_se == True:
            layers.append(
                CA(cnf.exp_channels)
            )
        else:
            layers.append(
                Spatial_SE(cnf.exp_channels, cnf.res)
            )
        
        layers.append(
            Stem(
                cnf.exp_channels,
                cnf.out_channels,
                kernel_size=1,
                activation=None))
        
        if cnf.use_softmax:
            layers.append(
                nn.Sigmoid())
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect == True:
            result = result + x
        return result

class MBlockConfig:
    def __init__(
        self,
        in_channels,
        exp_channels,
        out_channels,
        kernel_size,
        dilation,
        activation,
        downpool,
    ):
        self.in_channels = in_channels
        self.exp_channels = exp_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.downpool = downpool
        
class MBlock(nn.Module):
    def __init__(
        self,
        size,
        cnf: MBlockConfig,
        ):
        super(MBlock, self).__init__()
        self.connect = cnf.out_channels[0] == cnf.in_channels[0]
        H_layers = []
        L_layers= []
        # expand
        H_layers.append(
            nn.Sequential(
                Stem(
                    cnf.in_channels[0], 
                    cnf.exp_channels[0], 
                    kernel_size=1,
                    activation=cnf.activation
                ),
                Stem(
                    cnf.exp_channels[0],
                    cnf.exp_channels[0],
                    kernel_size=cnf.kernel_size,
                    stride=1,
                    groups=cnf.exp_channels[0],
                    dilation=cnf.dilation,
                    activation=cnf.activation,
                ),
                CA(cnf.exp_channels[0]),
                Stem(
                    cnf.exp_channels[0],
                    cnf.out_channels[0],
                    kernel_size=1,
                    activation=None)
            )
        )
        s = 2 if cnf.downpool else 1
        L_layers.append(
            nn.Sequential(
                Stem(
                    cnf.in_channels[1], 
                    cnf.exp_channels[1], 
                    kernel_size=1,
                    activation=cnf.activation
                ),
                Stem(
                    cnf.exp_channels[1],
                    cnf.exp_channels[1],
                    kernel_size=cnf.kernel_size,
                    stride=s,
                    groups=cnf.exp_channels[1],
                    dilation=cnf.dilation,
                    activation=cnf.activation,
                ),
                CA(cnf.exp_channels[1]),
                Stem(
                    cnf.exp_channels[1],
                    cnf.out_channels[1],
                    kernel_size=1,
                    activation=None)
            )
        )
        self.upppool = nn.Sequential(
            Stem(
                cnf.out_channels[1],
                cnf.out_channels[0],
                kernel_size=1,
                activation=cnf.activation
            ),
            nn.UpsamplingBilinear2d(size = size)
        )
        self.H_block = nn.Sequential(*H_layers)
        self.L_block = nn.Sequential(*L_layers)
        self.fusion = nn.Hardswish(inplace = True)
    def forward(self, x):
        x_H = x[0]
        x_L = x[1]
        x_H = self.H_block(x_H)
        x_L = self.L_block(x_L)
        if self.connect:
            x_H = self.fusion(x_H + self.upppool(x_L) + x[0])
        else:
            x_H = self.fusion(x_H + self.upppool(x_L))
        x[0] = x_H
        x[1] = x_L
        return x

class MobilePosNet(nn.Module):
    def __init__(
        self,
        BlockSetting: List[BlockConfig],
        MBlockSetting,
        # MBlockSetting: List[MBlockConfig],
        num_joints,
        unit_size,
        pmap_size,
        dropout = 0.2,
    ):
        super(MobilePosNet, self).__init__()
        
        self.num_joints = num_joints
        self.dropout = dropout
        self.pad = (unit_size[0] // 2, unit_size[1] // 2) # [32, 32] - [16, 16]
        padded_patch = (pmap_size[1] + 1, pmap_size[0] + 1) # [9, 7]
   
        layers = []
        Mlayers = []
        # building first layer
        first_output_channels = BlockSetting[0].in_channels
        layers.append(
            Stem(
                3,
                first_output_channels,
                kernel_size=3,
                stride=2,
                activation='HS'
            )          
        )

        # building stage1 other blocks
        for cnf in BlockSetting:
            layers.append(Block(padded_patch, cnf))
        
        # for cnf in MBlockSetting:
        #     Mlayers.append(MBlock(padded_patch, cnf))

        last_output_channel = BlockSetting[-1].out_channels
        last_channel = last_output_channel * 6
        self.Net = nn.Sequential(*layers)
        self.MNet = nn.Sequential(*Mlayers)
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(last_output_channel, last_channel, kernel_size = 1),
        #     nn.BatchNorm2d(last_channel),
        #     nn.Hardswish(inplace = True),
        #     nn.Conv2d(last_channel, num_joints, kernel_size = 1),
        #     nn.ReLU(inplace = True))

        self.classifier = nn.Sequential(
            nn.Conv2d(last_output_channel, 1280, kernel_size = 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(1280, num_joints, kernel_size = 1),
            nn.ReLU(inplace = True))
         
        self.regress = nn.Sequential(
            # nn.Conv2d(num_joints, num_joints, kernel_size = 2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU(inplace = True)
        )
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # nn.init.normal_(m.weight, 0, 0.01)
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            
    def forward(self, x):
        # print(x.shape)
        x = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
        # print(x.shape)
        x = self.Net(x)
        # mx = [x, x]
        # mx = self.MNet(mx)
        # print(x.shape)
        # [-1, num_joints, h, w]
        # x = mx[0]
        x = self.classifier(x)
        y = self.regress(x)
        x = x.flatten(2)
        y = y.flatten(2)
        x = F.normalize(x, p = 1, dim = 2)
        y = F.normalize(y, p = 1, dim = 2)
        # y = torch.zeros_like(x)
        # for i in range(self.num_joints):
        #     y[:, i, :] = self.regress(x[:,i,:])
       
        # y = F.normalize(y, p = 1, dim = 2)
        # print(x)
        # x = F.softmax(x, dim = 2)
        # x_adjusted = self.get_adjusted(x)
        return x, y
        # return x , x_adjusted

def _mobileposnet_conf(arch: str):
    # norm_layer = nn.BatchNorm2d(eps=0.01, momentum=0.01)
    stage_conf = BlockConfig
    Mstage_conf = MBlockConfig

    if arch == "_bnet_16ks":
        block_setting = [
            stage_conf(16, 16, 16, 3, 2, 1, 'RE', True),
            stage_conf(16, 72, 24, 3, 2, 1, 'RE', False),
            stage_conf(24, 88, 24, 3, 1, 1, 'RE', False),
            stage_conf(24, 96, 40, 5, 1, 1, 'HS', False),
            stage_conf(40, 240, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 240, 40, 5, 1, 1, 'HS', True),
            stage_conf(40, 120, 48, 5, 2, 1, 'HS', True),
            stage_conf(48, 144, 48, 5, 1, 1, 'HS', True),
            stage_conf(48, 288, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True) 
        ]
        Mstage_setting = None
    elif arch == "_bnet_16k":
        block_setting = [
            stage_conf(16, 16, 16, 3, 1, 1, 'RE', False),
            stage_conf(16, 64, 24, 3, 2, 1, 'RE', False),
            stage_conf(24, 72, 24, 3, 1, 1, 'RE', False),
            stage_conf(24, 72, 40, 5, 2, 1, 'RE', True),
            stage_conf(40, 120, 40, 5, 1, 1, 'RE', True),
            stage_conf(40, 120, 40, 5, 1, 1, 'RE', True),
            stage_conf(40, 240, 48, 3, 2, 1, 'HS', False),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            # stage_conf(40, 240, 40, 5, 1, 1, 'HS', True),
            # stage_conf(40, 120, 48, 5, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 5, 1, 1, 'HS', True),
            stage_conf(48, 288, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True) 
        ]
        Mstage_setting = None
    elif arch == "_bnet_16k_sse_new":
        block_setting = [
            stage_conf(32, 32, 16, 3, 1, 1, 'RE', False, 8),
            stage_conf(16, 64, 24, 3, 2, 1, 'RE', False, 4),
            stage_conf(24, 72, 24, 3, 1, 1, 'RE', False, 4),
            stage_conf(24, 144, 40, 5, 2, 1, 'RE', False,2),
            stage_conf(40, 120, 40, 5, 1, 1, 'RE', False, 2),
            stage_conf(40, 120, 40, 5, 1, 1, 'RE', False, 2),
            stage_conf(40, 240, 48, 5, 2, 1, 'HS', False),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 144, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 288, 48, 3, 1, 1, 'HS', True),
            stage_conf(48, 288, 96, 3, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True),
            stage_conf(96, 576, 96, 5, 1, 1, 'HS', True) 
        ]
        Mstage_setting = None
    
    elif arch == "_V2_":
        block_setting = [
            # 112-112
            stage_conf(32, 32, 16, 3, 1, 1, 'RE', False),
            # 112-112
            stage_conf(16, 96, 24, 3, 2, 1, 'RE', False),
            stage_conf(24, 144, 24, 3, 1, 1, 'RE', False),
            # stage_conf(24, 144, 24, 3, 1, 1, 'RE', False),
            # stage_conf(24, 144, 24, 3, 1, 1, 'RE', False),
            # 56-28
            stage_conf(24, 144, 32, 3, 2, 1, 'RE', False),
            # stage_conf(32, 192, 32, 3, 1, 1, 'RE', False),
            stage_conf(32, 192, 32, 3, 1, 1, 'RE', False),
            stage_conf(32, 192, 32, 3, 1, 1, 'RE', False),
            # 28-14
            stage_conf(32, 192, 64, 3, 2, 1, 'RE', False),
            stage_conf(64, 384, 64, 3, 1, 1, 'RE', False),
            stage_conf(64, 384, 64, 3, 1, 1, 'RE', False),
            stage_conf(64, 384, 64, 3, 1, 1, 'RE', False),

            stage_conf(64, 384, 96, 3, 1, 1, 'RE', False),
            stage_conf(96, 576, 96, 3, 1, 1, 'RE', False),
            stage_conf(96, 576, 96, 3, 1, 1, 'RE', False)
 
        ]
        Mstage_setting = None
    
    else:
        raise ValueError(f"Unsupported model type {arch}")
    return block_setting, Mstage_setting
    # return block_setting, Tblock_setting, last_channel

def get_pose_net(cfg, is_train):
    block_setting, Mstage_setting = _mobileposnet_conf(cfg.MODEL.NAME)
    # [6, 8]
    pmap_size = (cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[0], cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.PATCH_SIZE[1])
    model = MobilePosNet(block_setting, Mstage_setting, cfg.MODEL.NUM_JOINTS, cfg.MODEL.PATCH_SIZE, pmap_size)
    if is_train:
        model.init_weights()

    return model

    

             
        
        

        





        