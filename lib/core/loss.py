# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# The SimDR and SA-SimDR part:
# Written by Yanjie Li (lyj20@mails.tsinghua.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.inference import get_max_preds

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.split(1, 1)
        heatmaps_gt = target.split(1, 1)
        loss = 0
      
        for idx in range(num_joints):
            # print("JOINT {}".format(idx))
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            # print(target_weight[:, idx])
            # print(f"pred: {heatmap_pred} ")
            # print(f"-> {heatmap_pred.mul(target_weight[:, idx])}")
            # print(f"gt: {heatmap_gt}")
            # print(f"-> { heatmap_gt.mul(target_weight[:, idx])}")
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)
            print(f"loss: {loss}")

        return loss / num_joints

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # output = nn.Softmax(output, 2)
        heatmaps_pred = output.split(1, 1)
        heatmaps_gt = target.split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class KLDiscretLoss(nn.Module):
    def __init__(self, use_target_weight):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 2)
        self.Softmax = nn.Softmax(dim = 2)
        self.use_target_weight = use_target_weight
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output, target, target_weight):
        num_joints = output.size(1)
        output = self.LogSoftmax(output).split(1, 1)
        target = self.Softmax(target).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            # print(f"JOINT: {idx}")
            pred = output[idx].squeeze()
            gt = target[idx].squeeze()
            # print(target_weight[:, idx])
            # print(f"pred: {pred} ")
            # print(f"-> {pred.mul(target_weight[:, idx])}")
            # print(f"gt: {gt}")
            # print(f"-> {gt.mul(target_weight[:, idx])}")
            if self.use_target_weight:
                loss += self.criterion(
                    pred.mul(target_weight[:, idx]), 
                    gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(pred, gt)
            # print(f"loss: {loss}")
            
        return loss / num_joints

class SmoothL1Loss(nn.Module):
    """SmoothL1Loss loss.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self, use_target_weight):
        super().__init__()
        self.criterion = F.smooth_l1_loss
        self.use_target_weight = use_target_weight

    def forward(self, cfg, output, target, target_weight):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D]): Output regression.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        output, _ = get_max_preds(cfg, output.clone().detach().cpu().numpy())
        output = torch.from_numpy(output).cuda()
        # print("output = {}".format(output))
        # torch.from_numpy(get_preds(cfg, output.clone().detach().cpu().numpy())).cuda()
        # target, _ = get_max_preds(cfg, target.clone().detach().cpu().numpy())
        # target = torch.from_numpy(target).cuda()
        # print("target = {}".format(target))
        # torch.from_numpy(get_preds(cfg, target.clone().detach().cpu().numpy())).cuda()

        if self.use_target_weight:
            assert target_weight is not None
            loss = self.criterion(output * target_weight,
                                  target* target_weight)
        else:
            loss = self.criterion(output, target)
    
        return loss

class MultiLoss(nn.Module):
    def __init__(self, use_target_weight):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.criteron_1 = KLDiscretLoss(use_target_weight).cuda()
        self.criteron_2 = SmoothL1Loss(use_target_weight).cuda()
    def forward(self, cfg, output, output_coord,coord, target, target_weight):
        kldloss = self.criteron_1(output, target, target_weight)
        l1loss = self.criteron_2(cfg, output_coord, coord, target_weight)
        balance = 100000. #100000.
        loss = kldloss * balance  + l1loss
       
        kldloss = balance * kldloss

        # return (kldloss * 100000. + l1loss), kldloss, l1loss
        return loss, kldloss, l1loss
        
