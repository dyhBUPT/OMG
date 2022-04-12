"""
@Author: Du Yunhao
@Filename: validate.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/10 14:11
@Discription: validate
"""
import os
import torch
from torch import nn
from os.path import join
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from utils import *
from config import get_default_config
from datasets import CityFlowNLDataset
from models.model import MultiStreamNetwork

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
cfg = get_default_config()
method = 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss'
cfg.merge_from_file('configs/{}.yaml'.format(method))
val_num = 4
epoch = 580

print(datetime.now())

dataset_val = CityFlowNLDataset(
    cfg=cfg,
    transform=get_transforms(cfg, False),
    # transform_motion=get_transforms(cfg, False, 2),
    transform_motion=get_transforms_c4(cfg, False, 2),
    mode='val',
    val_num=val_num,
    multi_frame=True  # 多帧推理
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=cfg.TRAIN.BATCH_SIZE * 5,
    shuffle=False,
    num_workers=cfg.TRAIN.NUM_WORKERS
)
model = MultiStreamNetwork(cfg.MODEL)
model.cuda()
model = nn.DataParallel(model)
encoder_text_name = cfg.MODEL.ENCODER_TEXT
if 'roberta' in encoder_text_name:
    tokenizer = RobertaTokenizer.from_pretrained(encoder_text_name, local_files_only=True)
elif 'bert' in encoder_text_name:
    tokenizer = BertTokenizer.from_pretrained(encoder_text_name, local_files_only=True)
else:
    tokenizer = None

checkpoint = torch.load(join(cfg.DATA.ROOT_SAVE, '{}_fold{}/checkpoint_epoch{}.pth'.format(method, val_num, epoch)))
model.load_state_dict(checkpoint['state_dict'])

evaluate_v2(
    model,
    tokenizer,
    dataloader_val,
    '*',
    dim=cfg.MODEL.EMBED_DIM,
    dirSave='/data/dyh/checkpoints/AICity2022Track2/{}_fold{}'.format(method, val_num),
    suffix='_val',
    mode='v1'
)


print(datetime.now())