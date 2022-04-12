"""
@Author: Du Yunhao
@Filename: train.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:39
@Discription: Train
"""
import os
import sys
import time
import torch
import random
import argparse
from torch import nn
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from os.path import join, exists
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer

from utils import *
from config import get_default_config
from datasets import CityFlowNLDataset
from models.model import MultiStreamNetwork

# os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

parser = argparse.ArgumentParser(description='AICity2022Track2 Training')
parser.add_argument('--config', default='configs/Res50IBN+BERT_OMGb.yaml', type=str, help='config_file')
parser.add_argument('--valnum', default=4, type=int, help='val_num')
args = parser.parse_args()

cfg = get_default_config()
cfg.merge_from_file(args.config)

set_seed(cfg.MODEL.SEED)

sys.path.append(cfg.MODEL.PATH_CLIP)
import clip

val_num = args.valnum
dir_save = join(cfg.DATA.ROOT_SAVE, cfg.METHOD_NAME + '_fold%d' % val_num)
if not exists(dir_save):
    os.mkdir(dir_save)
sys.stdout = Logger(join(dir_save, 'log.txt'))

print(cfg)

dataset_train = CityFlowNLDataset(
    cfg=cfg,
    transform=get_transforms(cfg, True),
    # transform_motion=get_transforms(cfg, True, 2),
    transform_motion=get_transforms_c4(cfg, True, 2),
    mode='train',
    val_num=val_num
)
dataset_val = CityFlowNLDataset(
    cfg=cfg,
    transform=get_transforms(cfg, False),
    # transform_motion=get_transforms(cfg, False, 2),
    transform_motion=get_transforms_c4(cfg, False, 2),
    mode='val',
    val_num=val_num
)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.TRAIN.NUM_WORKERS,
    drop_last=True,
    # collate_fn=lambda x: x
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=cfg.TRAIN.BATCH_SIZE * 10,
    shuffle=False,
    num_workers=cfg.TRAIN.NUM_WORKERS
)

model = MultiStreamNetwork(cfg.MODEL)
model.cuda()
model = nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.BASE_LR)

checkpoints = torch.load(join(cfg.DATA.ROOT_SAVE, 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_fold4/checkpoint_epoch580.pth'))
model.load_state_dict(checkpoints['state_dict'], strict=False)
del checkpoints

encoder_text_name = cfg.MODEL.ENCODER_TEXT
if 'roberta' in encoder_text_name:
    tokenizer = RobertaTokenizer.from_pretrained(encoder_text_name, local_files_only=True)
elif 'bert' in encoder_text_name:
    tokenizer = BertTokenizer.from_pretrained(encoder_text_name, local_files_only=True)
else:
    tokenizer = None

global_step = 0
for epoch in range(cfg.TRAIN.MAX_EPOCH):
    # evaluate_v2(model, tokenizer, dataloader_val, epoch, dim=cfg.MODEL.EMBED_DIM)
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = get_lr(cfg.TRAIN, epoch)
    set_lr(optimizer, lr)
    progress = ProgressMeter(
        num_batches=len(dataloader_train) * cfg.TRAIN.ONE_EPOCH_REPEAT,
        meters=[batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    end = time.time()
    for _ in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        for idx, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            if 'CLIP' in encoder_text_name:
                tokens_text = clip.tokenize(batch['text'])
                tokens_text1 = clip.tokenize(batch['text1'])
                tokens_text2 = clip.tokenize(batch['text2'])
                tokens_text3 = clip.tokenize(batch['text3'])
                tokens_colorType = clip.tokenize(batch['color_type'])
                input_ = {
                    'crop': batch['crop'].cuda(),
                    'crop2': batch['crop2'].cuda(),
                    'motionmap': batch['motionmap'].cuda(),
                    'fg_motionmap': batch['fg_motionmap'].cuda(),
                    'text': tokens_text.cuda(),
                    'text1': tokens_text1.cuda(),
                    'text2': tokens_text2.cuda(),
                    'text3': tokens_text3.cuda(),
                    'color_type': tokens_colorType.cuda(),
                }
            else:
                tokens_text = get_tokens(tokenizer, batch['text'])
                tokens_text1 = get_tokens(tokenizer, batch['text1'])
                tokens_text2 = get_tokens(tokenizer, batch['text2'])
                tokens_text3 = get_tokens(tokenizer, batch['text3'])
                tokens_colorType = get_tokens(tokenizer, batch['color_type'])
                input_ = {
                    'crop': batch['crop'].cuda(),
                    'crop2': batch['crop2'].cuda(),
                    'motionmap': batch['motionmap'].cuda(),
                    'fg_motionmap': batch['fg_motionmap'].cuda(),
                    'text_input_ids': tokens_text['input_ids'].cuda(),
                    'text_attention_mask': tokens_text['attention_mask'].cuda(),
                    'text1_input_ids': tokens_text1['input_ids'].cuda(),
                    'text1_attention_mask': tokens_text1['attention_mask'].cuda(),
                    'text2_input_ids': tokens_text2['input_ids'].cuda(),
                    'text2_attention_mask': tokens_text2['attention_mask'].cuda(),
                    'text3_input_ids': tokens_text3['input_ids'].cuda(),
                    'text3_attention_mask': tokens_text3['attention_mask'].cuda(),
                    'color_type_input_ids': tokens_colorType['input_ids'].cuda(),
                    'color_type_attention_mask': tokens_colorType['attention_mask'].cuda(),
                }
            data_time.update(time.time() - end)
            features_crops, features_texts, tau, cls_logits = model(input_)
            tau = tau.mean().exp()
            loss = 0
            for features_crop in features_crops:
                for features_text in features_texts:
                    similarity_i2t = (tau * features_crop) @ features_text.t()
                    similarity_t2i = similarity_i2t.t()
                    infoNCE_i2t = F.cross_entropy(similarity_i2t, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                    infoNCE_t2i = F.cross_entropy(similarity_t2i, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                    loss += (infoNCE_i2t + infoNCE_t2i) / (2 * len(features_texts) * len(features_crops))
                    if cfg.MODEL.USE_TRIPLETLOSS:
                        roll = random.randint(1, cfg.TRAIN.BATCH_SIZE - 1)
                        triplet_i2t = F.triplet_margin_with_distance_loss(
                            features_crop, features_text, features_text.roll(roll, dims=0),
                            # distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                            distance_function=lambda x, y: F.pairwise_distance(x, y, p=2),
                            margin=cfg.MODEL.TRIPLETLOSS_MARGIN,
                        )
                        roll = random.randint(1, cfg.TRAIN.BATCH_SIZE - 1)
                        triplet_t2i = F.triplet_margin_with_distance_loss(
                            features_text, features_crop, features_crop.roll(roll, dims=0),
                            # distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
                            distance_function=lambda x, y: F.pairwise_distance(x, y, p=2),
                            margin=cfg.MODEL.TRIPLETLOSS_MARGIN,
                        )
                        loss += cfg.MODEL.TRIPLETLOSS_FACTOR * (triplet_i2t + triplet_t2i) \
                                / (2 * len(features_texts) * len(features_crops))
            for cls_logit in cls_logits:
                if cfg.MODEL.INSTANCELOSS:
                    loss += 0.5 * F.cross_entropy(cls_logit, batch['index'].long().cuda())
                elif cfg.MODEL.ID_LOSS:
                    loss += F.cross_entropy(cls_logit, batch['id'].long().cuda()) / len(cls_logits)
            losses.update(loss.item(), cfg.TRAIN.BATCH_SIZE)
            loss.backward()
            global_step += 1
            optimizer.step()
            batch_time.update(time.time() - end)
            if idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step % (len(dataloader_train) * cfg.TRAIN.ONE_EPOCH_REPEAT))
    if epoch % 10 == 0:
        path_save = join(dir_save, 'checkpoint_epoch%d.pth' % epoch)
        torch.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            path_save
        )
        evaluate_v2(model, tokenizer, dataloader_val, epoch, dim=cfg.MODEL.EMBED_DIM)
