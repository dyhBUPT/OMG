"""
@Author: Du Yunhao
@Filename: datasets.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:37
@Discription: Datasets
"""
import cv2
import json
import torch
import random
import numpy as np
from PIL import Image
from os.path import join
from random import uniform
from torch.utils.data import Dataset

class CityFlowNLDataset(Dataset):
    def __init__(self, cfg, transform, transform_motion=None, mode='train', val_num=4, middle_frame=False, multi_frame=False):
        assert mode in ('train', 'val')
        assert val_num in (0, 1, 2, 3, 4, -1)
        anno = dict()
        self.cfg = cfg
        cfg_data = cfg.DATA
        if mode == 'train':
            for num in {0, 1, 2, 3, 4} - {val_num}:
                anno_num = json.load(open(join(cfg_data.ROOT_DATA, cfg_data.DIR_ANNO, 'fold_%d.json' % num)))
                anno.update(anno_num)
        else:
            if val_num == -1: val_num = 4
            anno_num = json.load(open(join(cfg_data.ROOT_DATA, cfg_data.DIR_ANNO, 'fold_%d.json' % val_num)))
            anno.update(anno_num)
        self.ids = list(anno.keys())
        self.tracks = list(anno.values())
        self.indexs = list(range(len(self.ids)))
        self.mode = mode
        self.cfg_data = cfg_data
        self.transform = transform
        self.transform_motion = transform_motion
        self.middle_frame = middle_frame
        self.multi_frame = multi_frame
        self.dict_motionmaps = dict()
        self.dict_fg_motionmaps = dict()

    def __getitem__(self, item):
        index = self.indexs[item]
        uuid = self.ids[index]
        track = self.tracks[index]
        # 读取文本
        text, text_ov = '', ''
        if self.mode == 'train' and self.cfg_data.NLAUG and random.random() > 0.5:
            track_nl = random.sample(track['nl'] + track['nl_aug'], 3)
        else:
            track_nl = track['nl']
        for nl in track_nl:
            if len(nl.split(' ')) > 27:  # 截断过长句子，仅保留第一句
                nl = nl.split('.')[0] + '.'
            text += nl
        for nl in track['nl_other_views']:
            text_ov += nl
        # 模式
        if self.multi_frame:
            length = len(track['frames'])
            idx_frames = range(length)
            num_samples = 8
            if num_samples <= length:
                if self.mode == 'train':
                    sample_frames = sorted(random.sample(idx_frames, num_samples))
                else:
                    sample_frames = np.linspace(0, length - 1, num=num_samples).astype(int)
            else:
                sample_frames = list(idx_frames) + [idx_frames[-1]] * (num_samples - length)
            crop = list()
            # crop2 = list()
            for index_frame in sample_frames:
                path_img = join(self.cfg_data.ROOT_DATA, track['frames'][index_frame])
                img = Image.open(path_img).convert('RGB')
                box = track['boxes'][index_frame]
                crop_ = img.crop([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                crop_ = self.transform(crop_)
                crop.append(crop_)
                # 裁剪大图
                # W, H = img.size
                # x1 = max(0, box[0] - box[2])
                # y1 = max(0, box[1] - box[3])
                # x2 = min(W, box[0] + 2 * box[2])
                # y2 = min(H, box[1] + 2 * box[3])
                # crop2_ = img.crop([x1, y1, x2, y2])
                # crop2_ = self.transform(crop2_)
                # crop2.append(crop2_)
            index_frame = len(track['frames']) // 2
            path_img = join(self.cfg_data.ROOT_DATA, track['frames'][index_frame])
            img = Image.open(path_img).convert('RGB')
            box = track['boxes'][index_frame]
            W, H = img.size
            x1 = max(0, box[0] - box[2])
            y1 = max(0, box[1] - box[3])
            x2 = min(W, box[0] + 2 * box[2])
            y2 = min(H, box[1] + 2 * box[3])
            crop2 = img.crop([x1, y1, x2, y2])
            crop2 = self.transform(crop2)
        else:
            if self.middle_frame:
                index_frame = len(track['frames']) // 2
            elif self.mode == 'train':
                index_frame = int(uniform(0, len(track['frames'])))
            else:
                index_frame = len(track['frames']) // 2
            # 读取图像
            path_img = join(self.cfg_data.ROOT_DATA , track['frames'][index_frame])
            img = Image.open(path_img).convert('RGB')
            box = track['boxes'][index_frame]
            crop = img.crop([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            crop = self.transform(crop)
            # 裁剪大图
            W, H = img.size
            x1 = max(0, box[0] - box[2])
            y1 = max(0, box[1] - box[3])
            x2 = min(W, box[0] + 2 * box[2])
            y2 = min(H, box[1] + 2 * box[3])
            crop2 = img.crop([x1, y1, x2, y2])
            crop2 = self.transform(crop2)
        # 运动图
        motionmap = torch.zeros((1, 1), dtype=float)
        if self.cfg.MODEL.USE_MOTION:
            if uuid in self.dict_motionmaps:
                motionmap = self.dict_motionmaps[uuid]
            else:
                motionmap = Image.open(join(self.cfg_data.ROOT_DATA, self.cfg_data.MOTIONMAP_DIR, '{}.jpg'.format(uuid)))
                motionmap = self.transform_motion(motionmap)
                self.dict_motionmaps[uuid] = motionmap
        # 前景运动图
        fg_motionmap = torch.zeros((1, 1), dtype=float)
        if self.cfg.MODEL.USE_FgMOTION:
            if uuid in self.dict_fg_motionmaps:
                fg_motionmap = self.dict_fg_motionmaps[uuid]
            else:
                # fg_motionmap = Image.open(
                #     join(self.cfg_data.ROOT_DATA, self.cfg_data.FgMOTIONMAP_DIR, '{}.jpg'.format(uuid))
                # )
                fg_motionmap1 = Image.open(
                    join(self.cfg_data.ROOT_DATA, self.cfg_data.FgMOTIONMAP_DIR, '{}.jpg'.format(uuid))
                )
                fg_motionmap2 = Image.open(
                    join(self.cfg_data.ROOT_DATA, self.cfg_data.FgMOTIONMAP2_DIR, '{}.jpg'.format(uuid))
                ).convert('L')
                fg_motionmap = self.transform_motion(fg_motionmap1, fg_motionmap2)
                self.dict_fg_motionmaps[uuid] = fg_motionmap
        return {
            'index': index,
            'crop': crop,
            'crop2': crop2,
            'motionmap': motionmap,
            'fg_motionmap': fg_motionmap,
            'color_type': 'This is a {} {}'.format(track['color'], track['obj']),
            'text': text,
            'text_ov': text_ov,
            'text1': track_nl[0],
            'text2': track_nl[1],
            'text3': track_nl[2],
            'id': int(track['id'])
        }

    def __len__(self):
        return len(self.indexs)

class CityFlowNLDataset_TestText(Dataset):
    def __init__(self, cfg_data):
        text = json.load(open(join(cfg_data.ROOT_DATA, 'test_nlp_aug_color_obj.json')))
        self.ids = list(text.keys())
        self.texts = list(text.values())
        self.indexs = list(range(len(self.ids)))

    def __getitem__(self, item):
        index = self.indexs[item]
        uuid = self.ids[index]
        texts = self.texts[index]
        text, text_ov = '', ''
        for nl in texts['nl']:
            if len(nl.split(' ')) > 27:  # 截断过长句子，仅保留第一句
                nl = nl.split('.')[0] + '.'
            text += nl
        for nl in texts['nl_other_views']:
            if len(nl.split(' ')) > 27:  # 截断过长句子，仅保留第一句
                nl = nl.split('.')[0] + '.'
            text_ov += nl
        return {
            'uuid': uuid,
            'text': text,
            'text_ov': text_ov,
            'text1': texts['nl'][0],
            'text2': texts['nl'][1],
            'text3': texts['nl'][2],
            'color_type': 'This is a {} {}'.format(texts['color'], texts['obj']),
        }

    def __len__(self):
        return len(self.indexs)

class CityFlowDataset_TestTrack(Dataset):
    def __init__(self, cfg_data, transform, multi_frame):
        track = json.load(open(join(cfg_data.ROOT_DATA, 'test_tracks.json')))
        self.ids = list(track.keys())
        self.tracks = list(track.values())
        self.indexs = list(range(len(self.ids)))
        self.cfg_data = cfg_data
        self.transform = transform
        self.multi_frame = multi_frame

    def __getitem__(self, item):
        index = self.indexs[item]
        uuid = self.ids[index]
        track = self.tracks[index]
        if self.multi_frame:
            length = len(track['frames'])
            idx_frames =range(length)
            num_samples = 8
            if num_samples <= length:
                sample_frames = sorted(random.sample(idx_frames, num_samples))
            else:
                sample_frames = list(idx_frames) + [idx_frames[-1]] * (num_samples - length)
            crop = list()
            for index_frame in sample_frames:
                path_img = join(self.cfg_data.ROOT_DATA, track['frames'][index_frame])
                img = Image.open(path_img).convert('RGB')
                box = track['boxes'][index_frame]
                crop_ = img.crop([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                crop_ = self.transform(crop_)
                crop.append(crop_)
            index_frame = len(track['frames']) // 2
            path_img = join(self.cfg_data.ROOT_DATA, track['frames'][index_frame])
            img = Image.open(path_img).convert('RGB')
            box = track['boxes'][index_frame]
            W, H = img.size
            x1 = max(0, box[0] - box[2])
            y1 = max(0, box[1] - box[3])
            x2 = min(W, box[0] + 2 * box[2])
            y2 = min(H, box[1] + 2 * box[3])
            crop2 = img.crop([x1, y1, x2, y2])
            crop2 = self.transform(crop2)
        return {
            'uuid': uuid,
            'crop': crop,
            'crop2': crop2,
        }

    def __len__(self):
        return len(self.indexs)

if __name__ == '__main__':
    from config import get_default_config
    from utils import get_transforms
    cfg = get_default_config()
    transform = get_transforms(cfg, True)
    dataset = CityFlowNLDataset_Test(cfg.DATA)
