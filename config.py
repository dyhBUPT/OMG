"""
@Author: Du Yunhao
@Filename: config.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:27
@Discription: config
"""
from yacs.config import CfgNode as CN

_C = CN()
_C.METHOD_NAME = 'baseline'

_C.DATA = CN()
_C.DATA.ROOT_DATA = '/data/CityFlow_NL'
_C.DATA.SIZE = 288
_C.DATA.ROOT_SAVE = '/data/dyh/checkpoints/AICity2022Track2'
_C.DATA.MOTIONMAP_DIR = 'MotionMap/motion_map'
_C.DATA.FgMOTIONMAP_DIR = 'ForegroundMotionMap'
_C.DATA.FgMOTIONMAP2_DIR = 'ForegroundMotionMap2'
_C.DATA.DIR_ANNO = 'train_v1_fold5_nlpaug_id'
_C.DATA.NLAUG = False

_C.MODEL = CN()
_C.MODEL.FREEZE_TEXT_ENCODER = True
_C.MODEL.EMBED_DIM = 1024
_C.MODEL.NUM_CLASS = 2155  # 2498  # 样本对儿数量
_C.MODEL.NUM_IDS = 482  # 轨迹ID数量，ID 0~481
_C.MODEL.INSTANCELOSS = False
_C.MODEL.SEED = 3407
_C.MODEL.USE_MOTION = False
_C.MODEL.ENCODER_IMG = 'r50ibn'
_C.MODEL.ENCODER_TEXT = 'bert-base-uncased'
_C.MODEL.PATH_CLIP = '/home/dyh/project/CLIP'
_C.MODEL.USE_SINGLE_TEXT = False
_C.MODEL.USE_COLOR_TYPE = False
_C.MODEL.USE_BIGGER_IMAGE = False
_C.MODEL.USE_FgMOTION = False
_C.MODEL.USE_TRIPLETLOSS = False
_C.MODEL.TRIPLETLOSS_MARGIN = 0.3
_C.MODEL.TRIPLETLOSS_FACTOR = 1
_C.MODEL.SELF_ATTENTION_WEIGHTS = False
_C.MODEL.ID_LOSS = False

_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 40
_C.TRAIN.ONE_EPOCH_REPEAT = 30
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.PRINT_FREQ = 500
_C.TRAIN.BASE_LR = 0.01
_C.TRAIN.WARMUP_EPOCH = 10
_C.TRAIN.WARMUP_START_LR = 1e-5
_C.TRAIN.COSINE_END_LR = 0.0


def get_default_config():
    return _C.clone()