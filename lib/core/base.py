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
 
import time
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.inference import get_final_preds
from utils.transforms import flip_back, transform_preds


logger = logging.getLogger(__name__)


def trainer(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lk = AverageMeter()
    ls = AverageMeter()
    # acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, coord_target, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs, outputs_coord = model(input)
        coord_target = coord_target.cuda(non_blocking = True)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        # if isinstance(outputs, list):
        #     loss, kldloss, l1loss = criterion(outputs[0], target, target_weight)
        #     for output in outputs[1:]:
        #         loss, kldloss, l1loss += criterion(output, target, target_weight)
        # else:
        #     output = outputs
        loss, kldloss, l1loss = criterion(config, outputs, outputs_coord, coord_target, target, target_weight)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        lk.update(kldloss.item(), input.size(0))
        ls.update(l1loss.item(), input.size(0))

        # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                        #  target.detach().cpu().numpy())
        # acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            # msg = 'Epoch: [{0}][{1}/{2}]\t' \
            #       'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
            #       'Speed {speed:.1f} samples/s\t' \
            #       'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
            #       'Loss {loss.val:.5f} ({loss.avg:.5f})' .format(
            #           epoch, i, len(train_loader), batch_time=batch_time,
            #           speed=input.size(0)/batch_time.val,
            #           data_time=data_time, loss=losses)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\n'\
                  'KLDLoss {lk.val:.5f} ({lk.avg:.5f})\t'\
                  'SmL1Loss {ls.val:.5f} ({ls.avg:.5f})' .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, lk = lk, ls = ls)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
           
            writer_dict['train_global_steps'] = global_steps + 1

            


def validater(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    lk = AverageMeter()
    ls = AverageMeter()
    # acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, coord_target, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs, outputs_coord = model(input)
            # if isinstance(outputs, list):
            #     output = outputs[-1]
            # else:
            #     output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped,
                                           val_dataset.flip_pairs,
                                           config.MODEL.PATCH_SIZE,
                                           config.MODEL.HEATMAP_SIZE).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            coord_target = coord_target.cuda(non_blocking = True)

            loss, kldloss, l1loss = criterion(config, outputs, outputs_coord, coord_target, target, target_weight)
            # loss = criterion(config, outputs_coord, coord_target, target, target_weight)
            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            lk.update(kldloss.item(),  num_images)
            ls.update(l1loss.item(),  num_images)
            # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
            #                                  target.cpu().numpy())

            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            c = meta['center'].numpy()
            # print(f"c{c.shape}")
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            # j = meta['joints'].numpy()
            # j_r = j.copy()
            # for i in range(j_r.shape[0]):
                # j[i] = transform_preds(
            # j_r[i], c[i], s[i], [config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]])
            # print(f"j: \n{j[-1]}")

            preds, maxvals = get_final_preds(
                config, outputs_coord.clone().cpu().numpy(), c, s)

            # print(f"preds: \n{preds[-1]}")
            # coords_ = coords.copy()
            # for i in range(coords.shape[0]):
                # coords[i] = transform_preds(
            # coords_[i] * 2, c[i], s[i], [config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]])
            # print(f"coords: \n{coords[-1]}")

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\n'\
                      'KLDLoss {lk.val:.5f} ({lk.avg:.5f})\t'\
                      'SmL1Loss {ls.val:.5f} ({ls.avg:.5f})'  .format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, lk = lk, ls = ls)
                logger.info(msg)


        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
