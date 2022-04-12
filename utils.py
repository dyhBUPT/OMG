"""
@Author: Du Yunhao
@Filename: utils.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 10:36
@Discription: utils
"""
import sys
import math
import torch
import numpy as np
from os.path import join
from datetime import datetime
import torch.nn.functional as F
from torchvision import transforms
from ReRanking import *
from config import get_default_config

cfg = get_default_config()
sys.path.append(cfg.MODEL.PATH_CLIP)
import clip

def get_transforms(cfg, train, size=1):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATA.SIZE * size, scale=(0.8, 1)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.DATA.SIZE * size, cfg.DATA.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def get_transforms_c4(cfg, train, size=1):
    def transform_fn(img_c3, img_c1):
        img_c3 = transform_c3(img_c3)
        img_c1 = transform_c1(img_c1)
        return torch.cat((img_c3, img_c1), dim=0)
    if train:
        resize = transforms.RandomResizedCrop(cfg.DATA.SIZE * size, scale=(0.8, 1))
        rotation = transforms.RandomApply([transforms.RandomRotation(10)], p=0.5)
        transform_c3 = transforms.Compose([
            resize,
            rotation,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_c1 = transforms.Compose([
            resize,
            rotation,
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])
    else:
        transform_c3 = transforms.Compose([
            transforms.Resize((cfg.DATA.SIZE * size, cfg.DATA.SIZE * size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_c1 = transforms.Compose([
            transforms.Resize((cfg.DATA.SIZE * size, cfg.DATA.SIZE * size)),
            transforms.ToTensor(),
            transforms.Normalize(0, 1)
        ])
    return transform_fn

def get_lr(cfg_train, curr_epoch):
    if curr_epoch < cfg_train.WARMUP_EPOCH:
        return (
            cfg_train.WARMUP_START_LR
            + (cfg_train.BASE_LR - cfg_train.WARMUP_START_LR)
            * curr_epoch
            / cfg_train.WARMUP_EPOCH
        )
    else:
        return (
            cfg_train.COSINE_END_LR
            + (cfg_train.BASE_LR - cfg_train.COSINE_END_LR)
            * (
                math.cos(
                    math.pi * (curr_epoch - cfg_train.WARMUP_EPOCH) / (cfg_train.MAX_EPOCH - cfg_train.WARMUP_EPOCH)
                )
                + 1.0
            )
            * 0.5
        )

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def evaluate(model, tokenizer, dataloader, epoch, dirSave=None, suffix='_train'):
    model.eval()
    FeaturesCrop, FeaturesText = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            tokens = tokenizer.batch_encode_plus(
                batch['text'],
                padding='longest',
                return_tensors='pt'
            )
            if isinstance(batch['crop'], list):
                crop = [i.cuda() for i in batch['crop']]
            else:
                crop = batch['crop'].cuda()
            input_ = {
                'crop': crop,
                'text_input_ids': tokens['input_ids'].cuda(),
                'text_attention_mask': tokens['attention_mask'].cuda()
            }
            features_crop, features_text, tau, cls_logits = model(input_)
            FeaturesCrop.append(features_crop)
            FeaturesText.append(features_text)
    FeaturesCrop = torch.cat(FeaturesCrop, dim=0)
    FeaturesText = torch.cat(FeaturesText, dim=0)
    similarity = FeaturesText @ FeaturesCrop.t()
    # similarity = re_ranking_similarity(FeaturesText, FeaturesCrop, 20, 6, 0.3)

    gt = torch.arange(similarity.shape[0])
    r1, r5, r10, mrr = evaluate_recall_mrr(similarity, gt)
    print(datetime.now())
    print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format(epoch, r1, r5, r10, mrr))
    if dirSave:
        np.save(join(dirSave, 'FeaturesCrop{}.npy'.format(suffix)), FeaturesCrop.detach().cpu().numpy())
        np.save(join(dirSave, 'FeaturesText{}.npy'.format(suffix)), FeaturesText.detach().cpu().numpy())

def evaluate_v2(model, tokenizer, dataloader, epoch, dim=1024, dirSave=None, suffix='_train', mode='v1'):
    model.eval()
    FeaturesCrop, FeaturesText = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if isinstance(batch['crop'], list):
                crop = [i.cuda() for i in batch['crop']]
            else:
                crop = batch['crop'].cuda()
            if isinstance(batch['crop2'], list):
                crop2 = [i.cuda() for i in batch['crop']]
            else:
                crop2 = batch['crop2'].cuda()
            if tokenizer:
                tokens_text = get_tokens(tokenizer, batch['text'])
                tokens_text1 = get_tokens(tokenizer, batch['text1'])
                tokens_text2 = get_tokens(tokenizer, batch['text2'])
                tokens_text3 = get_tokens(tokenizer, batch['text3'])
                tokens_colorType = get_tokens(tokenizer, batch['color_type'])
                input_ = {
                    'crop': crop,
                    'crop2': crop2,
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
            else:
                tokens_text = clip.tokenize(batch['text'])
                tokens_text1 = clip.tokenize(batch['text1'])
                tokens_text2 = clip.tokenize(batch['text2'])
                tokens_text3 = clip.tokenize(batch['text3'])
                tokens_colorType = clip.tokenize(batch['color_type'])
                input_ = {
                    'crop': crop,
                    'crop2': crop2,
                    'motionmap': batch['motionmap'].cuda(),
                    'fg_motionmap': batch['fg_motionmap'].cuda(),
                    'text': tokens_text.cuda(),
                    'text1': tokens_text1.cuda(),
                    'text2': tokens_text2.cuda(),
                    'text3': tokens_text3.cuda(),
                    'color_type': tokens_colorType.cuda(),
                }
            features_crops, features_texts, tau, cls_logits = model(input_)
            features_crop = torch.cat(features_crops, dim=1)
            features_text = torch.cat(features_texts, dim=1)
            FeaturesCrop.append(features_crop)
            FeaturesText.append(features_text)
    FeaturesCrop = torch.cat(FeaturesCrop, dim=0)
    FeaturesText = torch.cat(FeaturesText, dim=0)
    assert FeaturesCrop.size(1) // dim == FeaturesCrop.size(1) / dim
    num_featuresImg = FeaturesCrop.size(1) // dim
    num_featuresText = FeaturesText.size(1) // dim
    similarity = []
    for i in range(num_featuresText):
        similarity_ = []
        for j in range(num_featuresImg):
            ft = FeaturesText[:, i * dim: (i + 1) * dim]
            fi = FeaturesCrop[:, j * dim: (j + 1) * dim]
            if mode == 'v1':
                similarity.append(ft @ fi.t())
            elif mode == 'v2':
                similarity_.append(ft @ fi.t())
        if similarity_:
            similarity.append(torch.stack(similarity_).max(dim=0)[0])
    similarity = torch.stack(similarity)
    similarity = torch.mean(similarity, dim=0)

    gt = torch.arange(similarity.shape[0])
    r1, r5, r10, mrr = evaluate_recall_mrr(similarity, gt)
    print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format(epoch, r1, r5, r10, mrr))
    if dirSave:
        np.save(join(dirSave, 'FeaturesCrop{}.npy'.format(suffix)), FeaturesCrop.detach().cpu().numpy())
        np.save(join(dirSave, 'FeaturesText{}.npy'.format(suffix)), FeaturesText.detach().cpu().numpy())


def evaluate_recall_mrr(sim, gt):
    if torch.is_tensor(sim):
        sim = sim.cpu().numpy()
    r1, r5, r10, mrr = 0, 0, 0, 0
    batch_size = gt.shape[0]
    for row, label in zip(sim, gt):
        idx = row.argsort()[::-1].tolist()
        rank = idx.index(label)
        if rank < 1:
            r1 += 1
        if rank < 5:
            r5 += 1
        if rank < 10:
            r10 += 1
        mrr += 1.0 / (rank + 1)
    return r1 / batch_size, r5 / batch_size, r10 / batch_size, mrr / batch_size

def evaluate_recall_mrr_reranking(sim1, sim2, gt, flag):
    if torch.is_tensor(sim1):
        sim1 = sim1.cpu().numpy()
    if torch.is_tensor(sim2):
        sim2 = sim2.cpu().numpy()
    r1, r5, r10, mrr = 0, 0, 0, 0
    batch_size = gt.shape[0]
    for row1, row2, label in zip(sim1, sim2, gt):
        idx1 = row1.argsort()[::-1]

        idx1_ =idx1[:flag]
        sim2_idx1_ = row2[idx1_]
        idx2 = sim2_idx1_.argsort()[::-1]
        idx = np.append(
            idx1_[idx2],
            idx1[flag:]
        ).tolist()

        # idx1_ =idx1[flag:]
        # sim2_idx1_ = row2[idx1_]
        # idx2 = sim2_idx1_.argsort()[::-1]
        # idx = np.append(
        #     idx1[:flag],
        #     idx1_[idx2]
        # ).tolist()

        rank = idx.index(label)
        if rank < 1:
            r1 += 1
        if rank < 5:
            r5 += 1
        if rank < 10:
            r10 += 1
        mrr += 1.0 / (rank + 1)
    return r1 / batch_size, r5 / batch_size, r10 / batch_size, mrr / batch_size

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_tokens(tokenizer, text):
    return tokenizer.batch_encode_plus(
        text,
        padding='longest',
        return_tensors='pt'
    )

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    val_num = 4
    dim = 1024
    root = '/data/dyh/checkpoints/AICity2022Track2'
    # method = 'Res50IBN+CLIP-B_OMG2a'
    method = 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss'
    fq = np.load('{}/{}_fold{}/FeaturesText_val.npy'.format(root, method, val_num))
    fg = np.load('{}/{}_fold{}/FeaturesCrop_val.npy'.format(root, method, val_num))
    fq, fg = torch.tensor(fq), torch.tensor(fg)
    print(fq.size())

    num_featuresText = fq.size(1) // dim
    fq = torch.mean(
        torch.stack([
            fq[:, i * dim: (i+1) * dim] for i in range(num_featuresText)
        ]),
        dim=0
    )

    num_featuresImage = fg.size(1) // dim
    fg = torch.mean(
        torch.stack([
            fg[:, i * dim: (i + 1) * dim] for i in range(num_featuresImage)
        ]),
        dim=0
    )

    print('numImg: {}, numText: {}'.format(num_featuresImage, num_featuresText))
    similarity = fq @ fg.t()
    # similarity = re_ranking_similarity(fq, fg, 200, 60, 0.8)
    similarity = - re_ranking(fq, fg, 20, 6, 0.95)

    gt = torch.arange(similarity.shape[0])
    r1, r5, r10, mrr = evaluate_recall_mrr(similarity, gt)
    # r1, r5, r10, mrr = evaluate_recall_mrr_reranking(similarity, similarity_, gt, 5)  # 5
    print(datetime.now())
    print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format('*', r1, r5, r10, mrr))