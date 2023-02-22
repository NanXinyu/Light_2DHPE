import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size,
        stride,
        dilation = 1,
        groups = 1,
        activation = 'ReLU',
        norm_layer = nn.BatchNorm2d,
    ):
        super(ConvModule, self).__init__()
        
        self.conv_layer = nn.Conv2d(
            inp, oup, kernel_size, stride = stride, padding = (kernel_size - 1) // 2 * dilation, 
            dilation = dilation, groups = groups, bias = False
        )
       
        self.norm_layer = norm_layer if norm_layer is None else norm_layer(oup, eps = 0.01, momentum = 0.01)

        if activation == 'ReLU':
            self.acti_layer = nn.ReLU(inplace = True)
        elif activation == 'Hardswish':
            self.acti_layer = nn.Hardswish(inplace = True)
        elif activation == 'Sigmoid':
            self.acti_layer = nn.Sigmoid()
        elif activation == 'Hardsigmoid':
            self.acti_layer = nn.Hardsigmoid()
        else:
            self.acti_layer = None

        
    def forward(self, x):
        x = self.conv_layer(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.acti_layer is not None:
            x = self.acti_layer(x)
        return x 

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class SpatialWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16):
        super(SpatialWeighting, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = ConvModule(
            channels,
            int(channels / ratio), 
            kernel_size=1, 
            stride=1,
            activation='ReLU',
            norm_layer=None)
        
        self.fc2 = ConvModule(
            int(channels / ratio), 
            channels, 
            kernel_size=1, 
            stride=1,
            activation='Sigmoid',
            norm_layer=None)
       
    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.fc1(out)
        out = self.fc2(out)
        return x * out


class CrossResolutionWeighting(nn.Module):

    def __init__(self,
                 channels,
                 ratio=16):
        super(CrossResolutionWeighting, self).__init__()
        
        self.channels = channels
        total_channel = sum(channels)
        self.fc1 = ConvModule(
            total_channel,
            int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            activation='ReLU',
            norm_layer=None
        )

        self.fc2 = ConvModule(
            int(total_channel / ratio),
            total_channel,
            kernel_size=1,
            stride=1,
            activation='Sigmoid',
            norm_layer=None
        )

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class ConditionalChannelWeighting(nn.Module):

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio):
        super().__init__()
        self.stride = stride

        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio)
            
        self.depthwise_convs = nn.ModuleList([
            ConvModule(
                channel,
                channel,
                kernel_size=3,
                stride=self.stride,
                groups=channel,
                activation=None) for channel in branch_channels])

        self.spatial_weighting = nn.ModuleList([
            SpatialWeighting(channels=channel, ratio=4)
            for channel in branch_channels
        ])

    def forward(self, x):
        # print("ConditionalChannelWeighting")
        # print(f"input dim:{len(x)}")
        # for i, s in enumerate(x):
        #     print(f"{i}:{s.shape}")
        x = [s.chunk(2, dim = 1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]
        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]
        
        out = [torch.cat([s1, s2], dim = 1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]
        # print(f"output dim: {len(out)}")
        # for i, s in enumerate(out):
            # print(f"{i}:{s.shape}")
        return out


class Stem(nn.Module):

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = ConvModule(
            in_channels,
            stem_channels,
            kernel_size=3,
            stride=2)

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=2,
                groups=branch_channels,
                activation=None),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1)
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=2,
            groups=mid_channels,
            activation=None)
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels
            if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1)

    def forward(self, x):
        # print("Stem")
        # print(f"input size:{x.shape}")
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim = 1)
        
        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((self.branch1(x1), x2), dim = 1)
        out = channel_shuffle(out, 2)
        # print(f"output size:{out.shape}")
        return out



class ShuffleUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super().__init__()
        self.stride = stride

        branch_features = out_channels // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    groups=in_channels,
                    activation=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                groups=branch_features,
                activation=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1))

    def forward(self, x):

        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim = 1)
        else:
            x1, x2 = x.chunk(2, dim = 1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        
        out = channel_shuffle(out, 2)
        
        return out

class Transition(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        input_size,
        final_stage,
        keep_scale = False) -> None:
        super().__init__()
        self.inp = inp
        self.oup = oup
        keep_scale = inp[0] == oup[0]
        self.relu = nn.ReLU(inplace = True)
        for i in range(len(inp)):
            for j in range(len(oup)):
                if i < j:
                    self.branch2 = ShuffleUnit(inp[i], oup[j], 2)
                    
                elif i == j:
                    if keep_scale or (len(inp) < len(oup)):
                        self.branch1 = nn.Identity()
                    else:
                        self.branch1 = ShuffleUnit(inp[i], oup[j], 2)

                elif i > j:
                    if keep_scale:
                        up_layer = nn.UpsamplingNearest2d(size = input_size) if final_stage == True else nn.UpsamplingNearest2d(scale_factor = 2)
                        self.branch2 = nn.Sequential(
                            nn.Conv2d(inp[i], oup[j], kernel_size = 1, stride = 1),
                            nn.BatchNorm2d(oup[j]),
                            nn.UpsamplingNearest2d(size = input_size) if final_stage else nn.UpsamplingNearest2d(scale_factor = 2))
                    else:
                        self.branch2 = nn.Identity()
    def forward(self, x):
        # print("Transition:")
        # print(f"input dim = {len(x)}")
        # for i, s in enumerate(x):
            # print(f"{i}:{s.shape}")
        out = []
        if len(self.oup) == 1:
            
            out.append(self.relu(self.branch1(x[0]) + self.branch2(x[1])))
        else:
            # out.append(ShuffleUnit(self.inp[0], self.oup[0], 2))
            out.append(self.branch1(x[0]))
            out.append(self.branch2(x[0]))
        # print(f"output dim = {len(out)}")
        # for i, s in enumerate(out):
            # print(f"{i}:{s.shape}")
        return out



class Stage(nn.Module):
    def __init__(
            self,
            num_blocks,
            in_channels,
            out_channels,
            reduce_ratio,
            pmap,
            final_stage,
            scale_keep=True,
            with_trans=True,
           
    ):
        super(Stage, self).__init__()
        

        self.in_channels = in_channels
        self.out_channels = out_channels
       
        self.with_trans = with_trans
        
        self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        
        if self.with_trans:
            self.fuse_layers = Transition(in_channels, out_channels, pmap, final_stage, scale_keep)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""
        out = self.layers(x)
        
        if self.with_trans:
            out = self.fuse_layers(x)
        return out

class Net(nn.Module):

    def __init__(self,
                 unit_size,
                 pmap,
                 num_joints,
                 last_channels = 768,
                 num_stages = 16,
                 channels = [
                    [32], 
                    [32, 64], # F
                    [32],     # T
                    [32, 64], # F

                    [64],     #
                    [64, 128], 
                    [64], 
                    [64, 128], 
                    [64], 
                    [64, 128], 
                    [64], 
                    [64, 128], 

                    [128], 
                    [128, 256],
                    [128], 
                    [128, 256], 
                    [128]],

                #  [
                #     [40], 
                #     [40, 80], 
                #     [80], 
                #     [80, 160], 
                #     [160], 
                #     [160, 320], 
                #     [160]],
                 num_blocks = [
                    1, 
                    2, 
                    1,
                    2,

                    1,
                    2,
                    1,
                    2,
                    1,
                    2,
                    1,
                    2,

                    1,
                    2,
                    1,
                    2],
                 reduce_ratio = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                 keep_scale = [
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True]):
        super().__init__()
        self.pmap = pmap
        self.pad = (unit_size[0] // 2, unit_size[1] // 2)
        self.stem = Stem(
            in_channels=3,
            stem_channels=64,
            out_channels=64,
            expand_ratio=1)
        
        self.first_layer = ConvModule(64, channels[0][0], 3, 1)
        
        stages = []
        for i in range(num_stages):
            if i == num_stages-1 or i == num_stages - 2 or i == num_stages -3:
                final_stage =True
            else:
                final_stage = False
            stages.append(Stage(num_blocks[i],
            channels[i],
            channels[i+1],
            reduce_ratio=reduce_ratio[i],
            pmap=pmap,
            final_stage=final_stage,
            scale_keep=keep_scale[i]))
            
        self.stages = nn.Sequential(*stages)

        self.last_layer = nn.Sequential(
            # nn.Conv2d(channels[-1][0], last_channels, 1),
            # nn.BatchNorm2d(last_channels),
            # nn.ReLU(inplace = True),
            # nn.Conv2d(last_channels, num_joints, 1),
            nn.Conv2d(channels[-1][0], num_joints, 1),
            nn.ReLU(inplace = True)           
        )

        self.upsample = nn.Sequential(
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

    def forward(self, x):
        """Forward function."""
        x = F.pad(x, (self.pad[0], self.pad[0], self.pad[1], self.pad[1]))
        x = self.stem(x)
        x = self.first_layer(x)
        x = self.stages([x])
        x = self.last_layer(x[0])
        y = self.upsample(x)
        x = x.flatten(2)
        y = y.flatten(2)
        x = F.normalize(x, p = 1, dim = 2)
        y = F.normalize(y, p = 1, dim = 2)

        return x, y

def get_pose_net(cfg, is_train):
    unit_size = cfg.MODEL.PATCH_SIZE
    pmap_size = (cfg.MODEL.IMAGE_SIZE[1] // cfg.MODEL.PATCH_SIZE[1] + 1, cfg.MODEL.IMAGE_SIZE[0] // cfg.MODEL.PATCH_SIZE[0] + 1)
    model = Net(unit_size, pmap_size, 17)
    if is_train:
        model.init_weights()

    return model