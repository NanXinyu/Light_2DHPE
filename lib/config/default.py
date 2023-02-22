from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 8
_C.PRINT_FREQ = 100
_C.AUTO_RESUME = True
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# MODEL
_C.MODEL = CN()
_C.MODEL.NAME =  "_V2_" #"_prv_mini" #"_distributepos_gaint0.2" # "_pmap_gaint"
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.IMAGE_SIZE = [192, 256]
# _C.MODEL.PATCH_RANGE = 16
_C.MODEL.PATCH_SIZE = [16, 16]
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.AUX_ALPHA = 0.00001

# LOSS
_C.LOSS = CN()
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
_C.LOSS.USE_TARGET_WEIGHT = True

# DATASET
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco'
_C.DATASET.TRAIN_SET = 'train2017'
_C.DATASET.TEST_SET = 'val2017'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.TRAIN_RATIO = 1.0
_C.DATASET.TEST_RATIO = 1.0

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25 # 0.35
_C.DATASET.ROT_FACTOR = 30
_C.DATASET.PROB_HALF_BODY = 0.3
_C.DATASET.NUM_JOINTS_HALF_BODY = 8
_C.DATASET.COLOR_RGB = True

# TRAIN
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 120]
_C.TRAIN.LR = 0.001


_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 250

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.BATCH_SIZE_PER_GPU = 128
_C.TRAIN.SHUFFLE = True

# TEST
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
# Test Model Epoch
_C.TEST.FLIP_TEST = False
_C.TEST.POST_PROCESS = False
_C.TEST.SHIFT_HEATMAP = False

_C.TEST.USE_GT_BBOX = True
_C.TEST.BLUR_KERNEL = 11

# nms
_C.TEST.IMAGE_THRE = 0.0
_C.TEST.NMS_THRE = 1.0
_C.TEST.SOFT_NMS = False
_C.TEST.OKS_THRE = 0.9
_C.TEST.IN_VIS_THRE = 0.02 #0.001
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.BBOX_THRE = 1.0
_C.TEST.MODEL_FILE = ''

# PCKH
_C.TEST.PCKH_THRE = 0.5

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False

def update_config(cfg):
    # RPR-Pose/
    cfg.CUR_DIR = osp.dirname(osp.abspath(__file__))
    cfg.ROOT_DIR = osp.dirname(osp.dirname(cfg.CUR_DIR))

    # PRP-Pose/output
    cfg.OUTPUT_DIR = osp.join(cfg.ROOT_DIR, 'output')
    
    # PRP-Pose/output/log
    cfg.LOG_DIR = osp.join(cfg.ROOT_DIR, 'log')

    cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, 'data')

    cfg.DATASET.ROOT =  '/root/repo/MHPE/data/coco/'

    cfg.TEST_FILE = os.path.join(
        cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )
    
    cfg.freeze()

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)