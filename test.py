"""
@Author: Du Yunhao
@Filename: test.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/18 14:45
@Discription: test
"""
import os
import json
import torch
from torch import nn
from os.path import join
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from utils import *
from datasets import *
from ReRanking import re_ranking
from config import get_default_config
from models.model import MultiStreamNetwork

def Step1_FeaturesExtraction(epoch):
    dataset_text = CityFlowNLDataset_TestText(
        cfg_data=cfg.DATA
    )
    dataset_crop = CityFlowDataset_TestTrack(
        cfg_data=cfg.DATA,
        transform=get_transforms(cfg, False),
        multi_frame=True
    )
    dataloader_text = DataLoader(
        dataset_text,
        batch_size=cfg.TRAIN.BATCH_SIZE * 5,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )
    dataloader_crop = DataLoader(
        dataset_crop,
        batch_size=cfg.TRAIN.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS
    )

    model = MultiStreamNetwork(cfg.MODEL)
    model.cuda()
    model = nn.DataParallel(model)
    encoder_text_name = cfg.MODEL.ENCODER_TEXT
    if 'roberta' in encoder_text_name:
        tokenizer = RobertaTokenizer.from_pretrained(encoder_text_name)
    elif 'bert' in encoder_text_name:
        tokenizer = BertTokenizer.from_pretrained(encoder_text_name)
    else:
        tokenizer = None
    checkpoint = torch.load(join(cfg.DATA.ROOT_SAVE, method_name, 'checkpoint_epoch{}.pth'.format(epoch)))
    model.load_state_dict(checkpoint['state_dict'], strict=False)


    Features_Text, UUIDs_Text = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader_text):
            if 'CLIP' in encoder_text_name:
                tokens_text = clip.tokenize(batch['text'])
                tokens_text1 = clip.tokenize(batch['text1'])
                tokens_text2 = clip.tokenize(batch['text2'])
                tokens_text3 = clip.tokenize(batch['text3'])
                tokens_colorType = clip.tokenize(batch['color_type'])
                input_ = {
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
            features_text = torch.cat([
                model.module.encode_text(input_, 'text'),
                model.module.encode_text(input_, 'text1'),
                model.module.encode_text(input_, 'text2'),
                model.module.encode_text(input_, 'text3'),
                model.module.encode_text(input_, 'color_type'),
            ], dim=1)
            Features_Text.append(features_text)
            UUIDs_Text.extend(batch['uuid'])
    FeaturesText = torch.cat(Features_Text, dim=0)
    np.save(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_FeaturesText_{}.npy'.format(epoch)), FeaturesText.detach().cpu().numpy())
    json.dump(UUIDs_Text, open(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_uuidText.json'), 'w'))

    Features_Img, UUIDs_Img = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader_crop):
            if isinstance(batch['crop'], list):
                crop = [i.cuda() for i in batch['crop']]
            else:
                crop = batch['crop'].cuda()
            if isinstance(batch['crop2'], list):
                crop2 = [i.cuda() for i in batch['crop']]
            else:
                crop2 = batch['crop2'].cuda()
            features_img = torch.cat([
                model.module.encode_img({'crop': crop}, 'crop'),
                model.module.encode_img({'crop2': crop2}, 'crop2'),
            ], dim=1)
            Features_Img.append(features_img)
            UUIDs_Img.extend(batch['uuid'])
    FeaturesImg = torch.cat(Features_Img, dim=0)
    np.save(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_FeaturesImg_{}.npy'.format(epoch)), FeaturesImg.detach().cpu().numpy())
    json.dump(UUIDs_Img, open(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_uuidImg.json'), 'w'))

def Step2_test(rr=False, mode='v1'):
    dim = cfg.MODEL.EMBED_DIM
    FeaturesText = np.load(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_FeaturesText.npy'))
    uuidText = np.array(json.load(open(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_uuidText.json'), 'r')))
    FeaturesImg = np.load(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_FeaturesImg.npy'))
    uuidImg = np.array(json.load(open(join(cfg.DATA.ROOT_SAVE, method_name, 'Test_uuidImg.json'), 'r')))
    results = dict()
    num_featuresImg = FeaturesImg.shape[1] // dim
    num_featuresText = FeaturesText.shape[1] // dim
    similarity = []
    print('numImg: {}, numText: {}'.format(num_featuresImg, num_featuresText))
    for i in range(num_featuresText):
        similarity_ = []
        for j in range(num_featuresImg):
            ft = FeaturesText[:, i * dim: (i + 1) * dim]
            fi = FeaturesImg[:, j * dim: (j + 1) * dim]
            if rr:
                sim = -re_ranking(torch.tensor(ft), torch.tensor(fi), 20, 6, 0.95)
            else:
                sim = ft @ fi.T
            if mode == 'v1':
                similarity.append(sim)
            elif mode == 'v2':
                similarity_.append(torch.tensor(sim))
        if similarity_:
            similarity.append(torch.stack(similarity_).max(dim=0)[0].numpy())

    similarity = np.stack(similarity)
    if similarity.shape[0] > 1:
        similarity = np.mean(similarity, axis=0)
    else:
        similarity = similarity[0]

    for i, sim in enumerate(similarity):
        idx = sim.argsort()[::-1]
        print(sim[idx])
        results[uuidText[i]] = uuidImg[idx].tolist()
    json.dump(
        results,
        open(join(
            cfg.DATA.ROOT_SAVE,
            method_name,
            'Test_results_rr_{}.json'.format(mode) if rr else 'Test_results_{}.json'.format(mode)
        ), 'w'),
        indent=2
    )


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cfg = get_default_config()
    cfg.merge_from_file('configs/Swin-B+CLIP-B_OMG2a_NLAug_IDLoss.yaml')
    method_name = 'Swin-B+CLIP-B_OMG2a_NLAug_IDLoss_fold4'

    '''1.特征提取'''
    Step1_FeaturesExtraction(epoch=580)
    '''2.计算结果'''
    # Step2_test(
    #     rr=False,
    #     mode='v1'
    # )