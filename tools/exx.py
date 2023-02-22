import torch
import torch.nn as nn
import numpy as np

import _init_paths
from core.model import get_pose_net 
from config import cfg
from einops import rearrange
from core.loss import JointsMSELoss, KLDiscretLoss, SmoothL1Loss, MultiLoss

# from thop import profile
from torchstat import stat
from thop import profile
# import torchsummary as summary

from torchsummary import summary
import os
import torch.nn.functional as F
from core.inference import get_max_preds
# from core.evaluate import accuracy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]
        
        padded_size = self.image_size + self.patch_size
        
        nw = padded_size[0] // self.patch_size[0]
        nh = padded_size[1] // self.patch_size[1]

        patch_target = np.zeros((self.num_joints,
                           nh, nw),
                           dtype=np.float32)
        coord_target = np.zeros((self.num_joints, 2), dtype = np.float32)
        

        for joint_id in range(self.num_joints):
            mu_x = joints[joint_id][0] 
            mu_y = joints[joint_id][1]
            x0 = mu_x + self.patch_size[0] // 2
            y0 = mu_y + self.patch_size[1] // 2
            ul = [int(x0 - self.sigma[1] * 3), int(y0 - self.sigma[0] * 3)]
            br = [int(x0 + self.sigma[1] * 3 + 1), int(y0 + self.sigma[1] * 3 + 1)]
            
            if ul[0] >= padded_size[1] or ul[1] >= padded_size[0] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue
            # print(padded_size)
            # # Generate gaussian
            x = np.arange(0, padded_size[0], 1, np.float32)
            # print(x.shape)
            y = np.arange(0, padded_size[1], 1, np.float32)
            y = y[:, np.newaxis]
            # print(y.shape)
            g = np.exp(- (((x - x0) / self.sigma[1])** 2 + ((y - y0) / self.sigma[0]) ** 2) / 2) / ( 2 * np.pi * self.sigma[0] * self.sigma[1])
            print(g.sum())
           
            v = target_weight[joint_id]
            if v > 0.5:
                patch_target[joint_id][:] = rearrange(g, '(nh ph) (nw pw)-> nh nw ph pw', \
                nh = nh, ph = self.patch_size[1], 
                nw = nw, pw = self.patch_size[0]).sum(axis = (2, 3))
                # # print(f"nh:{nh}, nw:{nw}")
                m = F.MaxPool2d(16)
                patch_target[joint_id][:] = m(g)
                coord_target[joint_id][0] = mu_x
                coord_target[joint_id][1] = mu_y
            
            
            
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)
            
        return coord_target, patch_target, target_weight

class TestConfig:
        def __init__(
        self,
        num_joints,
        image_size,
        patch_size,
        joints_weight,
        use_different_joints_weight,
        ):
          self.num_joints = num_joints
          self.image_size = image_size
          self.patch_size = np.array(patch_size)
          self.joints_weight = joints_weight
          self.sigma = np.array([(patch_size[0])/6, (patch_size[1])/6]) # [patch_size[0]/6, patch_size[1]/6]
          self.use_different_joints_weight = use_different_joints_weight

def get_preds(cfg, patch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, patch_size])
    '''
    assert isinstance(patch_heatmaps, np.ndarray), \
        'patch_heatmaps should be numpy.ndarray'
    assert patch_heatmaps.ndim == 3, 'patch_images should be 4-ndim'

    batch_size = patch_heatmaps.shape[0] # -1
    num_joints = patch_heatmaps.shape[1] # 17
    patch_size = patch_heatmaps.shape[2] # 25
    
    width = cfg.MODEL.IMAGE_SIZE[0] + cfg.MODEL.PATCH_SIZE[1]
    index = np.zeros((batch_size, num_joints, 1)).astype(np.float32)
    # print(np.argmax(patch_heatmaps, axis = 2, keepdim = True).shape)
    index = np.argmax(patch_heatmaps, axis = 2)
    # print(patch_heatmaps[:, :].shape)
    
    preds = np.zeros((batch_size, num_joints, 2)).astype(np.float32)
    preds[:, :, 0] = (index % width) - cfg.MODEL.PATCH_SIZE[1] // 2
    preds[:, :, 1] = (index // width) - cfg.MODEL.PATCH_SIZE[1] // 2
    # print(preds)
    maxvals = np.amax(patch_heatmaps, 2).reshape((batch_size, num_joints, 1))
    # print(maxvals)
    # pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    # pred_mask = pred_mask.astype(np.float32)

    return preds, maxvals
if __name__ == '__main__':
    
    
    # x = torch.rand(1, 3, 5, 5).cuda()
    # _, _, h, w = x.shape
    # print(x)
    # x = rearrange(x, 'b c h w -> b (h w) c')
    # print(x)
    # x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
    # print(x)
    # x = torch.rand(1, 3, 256, 192).cuda()
    # model = get_pose_net(cfg, True).cuda()
    # output, r= model(x)
    # print(output.shape)
    # print(r.shape)
    # inputs = torch.rand((1, 3, 256, 192))
    # model = get_pose_net(cfg, is_train=True)
    # summary(model)
    # print(x.max())
    # summary.summary(model, (3, 256, 192))
    # stat(get_pose_net(cfg, True), (3, 256, 192))
    # dummy_input = torch.randn(1, 3, 256, 192)
    # model_ = get_pose_net(cfg, True)
    # flops, params = profile(model_, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    testdata = TestConfig(17, [192, 256], [16, 16], 1, True)
    joints = torch.rand(17, 2)
    joints[:, 0] = joints[:, 0] * 192
    joints[:, 1] = joints[:, 1] * 256
    joints_vis = torch.ones(17, 1)
    coord, target, target_weight= generate_target(testdata, joints.detach().numpy(), joints_vis)
    coord = torch.from_numpy(coord).unsqueeze(0)
    target = torch.from_numpy(target).unsqueeze(0)
    print(target.shape)
    
    target_weight = torch.from_numpy(target_weight).unsqueeze(0)
    uppool = nn.Upsample(scale_factor=16, mode='bilinear')# nn.UpsamplingNearest2d(scale_factor  = 16)
    target = uppool(target)
    print(target.shape)
    target = F.normalize(target.flatten(2), p = 1, dim = 2)
    print(target.shape)
    
    
    # cri = MultiLoss(True)
    # # l1, l2, l3 = cri(cfg, output, coord.cuda(), target.cuda(), target_weight.cuda())
    # # print("Multi Loss: {}".format(l1))
    
    

    preds, maxvals = get_preds(cfg, target.detach().numpy())
    for i in range(preds.shape[1]):
        print(f"JOINT{i}")
        print("preds:")
        print(preds[:,i,:])
        print("coords:")
        print(coord[:,i,:])
        print("maxvals:")
        print(maxvals[:,i,:])
    
    # criteron_ = KLDiscretLoss(True)
    # loss_ = criteron_(output, target.cuda(), target_weight.cuda())
    # print("KLDiscretLoss: {}".format(loss_))
    criteron = SmoothL1Loss(True)
    loss = criteron(cfg, target, coord.cuda(), target_weight.cuda())
    print("SmoothL1Loss: {}".format(loss))
    # acc, avg_acc, cnt, pred = accuracy(cfg, output.detach().cpu().numpy(),
    #                                  coord.detach().cpu().numpy())
    # print(acc)
    # # print("total")

    # print(target.sum(axis = 1))
    # coord = torch.from_numpy(coord).unsqueeze(0)
    # target = torch.from_numpy(target).unsqueeze(0)
    # target_weight = torch.from_numpy(target_weight).unsqueeze(0)
    

    

    # print("GET PREDS:")
    # print("limited")
    # preds = get_final_preds(cfg, target.detach().numpy())
    
    # print("coord:")
    # print(coord)
    # print("PRED")
    # preds = get_final_preds(cfg, output.detach().numpy())
    # print(preds)
    
    
    
    