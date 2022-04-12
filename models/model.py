"""
@Author: Du Yunhao
@Filename: model.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 11:05
@Discription: model
"""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import RobertaModel
from torchvision.models import resnet50
from config import get_default_config
from models.SENet import se_resnext50_32x4d
from models.Res50IBN import build_resnet_backbone
from models.SwinTransformer import SwinTransformer

cfg = get_default_config()
sys.path.append(cfg.MODEL.PATH_CLIP)
import clip

class mlp_text(nn.Module):
    def __init__(self, dim_text, dim_embedding):
        super(mlp_text, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_text, dim_text),
            nn.ReLU(),
            nn.Linear(dim_text, dim_embedding)
        )
        
    def forward(self, x):
        return self.mlp(x)

class MultiStreamNetwork(nn.Module):
    def __init__(self, cfg_model, mode='train'):
        super(MultiStreamNetwork, self).__init__()
        assert mode in ('train', 'test')
        self.mode = mode
        self.cfg_model = cfg_model
        dim_embedding = cfg_model.EMBED_DIM

        if 'roberta' in cfg_model.ENCODER_TEXT:
            dim_text = 1024
        elif 'bert' in cfg_model.ENCODER_TEXT:
            dim_text = 768
        elif cfg_model.ENCODER_TEXT == 'CLIP_ViT-B/32':
            dim_text = 512
        elif cfg_model.ENCODER_TEXT == 'CLIP_ViT-L/14':
            dim_text = 768

        if cfg_model.ENCODER_IMG == 'CLIP_ViT-B/32':
            dim_img = 512
        elif cfg_model.ENCODER_IMG == 'CLIP_ViT-L/14':
            dim_img = 768
        elif cfg_model.ENCODER_IMG == 'SwinTransformer':
            dim_img = 1024
        else:
            dim_img = 2048

        if cfg_model.ENCODER_IMG == 'SwinTransformer':
            self.fc_img = nn.Linear(dim_img, dim_embedding)
        else:
            self.fc_img = nn.Sequential(
                nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
                nn.AdaptiveAvgPool2d(1)  # 平均池化
            )

        self.fc_text = mlp_text(dim_text, dim_embedding)
        if cfg_model.USE_SINGLE_TEXT:
            self.fc_text1 = mlp_text(dim_text, dim_embedding)
            self.fc_text2 = mlp_text(dim_text, dim_embedding)
            self.fc_text3 = mlp_text(dim_text, dim_embedding)
        if cfg_model.USE_COLOR_TYPE:
            self.fc_color_type = mlp_text(dim_text, dim_embedding)

        self.tau = nn.Parameter(torch.ones(1), requires_grad=True)

        if cfg_model.INSTANCELOSS:
            self.fc_share = nn.Sequential(
                nn.Linear(dim_embedding, dim_embedding),
                nn.BatchNorm1d(dim_embedding),
                nn.ReLU(),
                nn.Linear(dim_embedding, cfg_model.NUM_CLASS)
            )

        if cfg_model.USE_MOTION:
            self.encoder_motion = self.get_img_encoder(name=encoder_img)
            self.fc_motion = nn.Sequential(
                nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
                nn.AdaptiveMaxPool2d(1)  # 最大池化
            )
            self.fc_merge = nn.Sequential(
                nn.Linear(2 * dim_embedding, dim_embedding),
                nn.ReLU(),
                nn.Linear(dim_embedding, dim_embedding)
            )

        if 'CLIP' in cfg_model.ENCODER_TEXT:
            self.clip, self.clip_preprocess = clip.load(cfg_model.ENCODER_TEXT.split('_')[1])
            for key, value in self.clip.named_parameters():
                value.requires_grad = False

        encoder_img = cfg_model.ENCODER_IMG
        self.encoder_img = self.get_img_encoder(name=encoder_img)
        self.encoder_text = self.get_text_encoder(cfg_model)

        if cfg_model.USE_BIGGER_IMAGE:
            self.encoder_img2 = self.get_img_encoder(name=encoder_img)
            # self.encoder_img2 = self.encoder_img
            if cfg_model.ENCODER_IMG == 'SwinTransformer':
                self.fc_img2 = nn.Linear(dim_img, dim_embedding)
            else:
                self.fc_img2 = nn.Sequential(
                    nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1)  # 平均池化
                )

        if cfg_model.USE_FgMOTION:
            # self.encoder_fgmotion = self.get_img_encoder(name=encoder_img, in_channels=3)
            self.encoder_fgmotion = self.get_img_encoder(name=encoder_img, in_channels=4)
            self.fc_fgmotion = nn.Sequential(
                nn.Conv2d(dim_img, dim_embedding, kernel_size=1),
                nn.AdaptiveAvgPool2d(1)  # 平均池化
            )

        if cfg_model.SELF_ATTENTION_WEIGHTS:
            self.attention_img = nn.TransformerEncoderLayer(
                d_model=dim_embedding, nhead=8, dim_feedforward=dim_embedding, dropout=0.1, activation='relu'
            )
            self.attention_text = nn.TransformerEncoderLayer(
                d_model=dim_embedding, nhead=8, dim_feedforward=dim_embedding, dropout=0.1, activation='relu'
            )

        if cfg_model.ID_LOSS:
            self.classifier_img = nn.Sequential(
                nn.Linear(dim_embedding, dim_embedding),
                nn.ReLU(),
                nn.Linear(dim_embedding, cfg_model.NUM_IDS)
            )
            self.classifier_text = nn.Sequential(
                nn.Linear(dim_embedding, dim_embedding),
                nn.ReLU(),
                nn.Linear(dim_embedding, cfg_model.NUM_IDS)
            )

    def get_img_encoder(self, name='r50', in_channels=3):
        assert name in ('r50', 'se50', 'r50ibn', 'r101ibn', 'CLIP_ViT-B/32', 'SwinTransformer')
        if name == 'resnet50':
            encoder = resnet50(pretrained=True)
            features = list(encoder.children())[:-2]  # 去掉GAP和FC
            encoder = nn.Sequential(*features)
        elif name == 'se50':
            encoder = se_resnext50_32x4d(pretrained=None)
            ckpt = torch.load('/data/dyh/checkpoints/AICity2022Track2/motion_SE_NOCLS_nlpaug_288.pth')
            state_dict = dict()
            for key, value in ckpt['state_dict'].items():
                if 'vis_backbone.' in key:
                    state_dict[key.replace('module.vis_backbone.', '')] = value
            encoder.load_state_dict(state_dict)
        elif name == 'r50ibn':
            encoder = build_resnet_backbone('50x', pretrain=(self.mode=='train'), in_channels=in_channels)
        elif name =='r101ibn':
            encoder = build_resnet_backbone('101x', pretrain=(self.mode=='train'), in_channels=in_channels)
        elif 'CLIP' in name:
            assert self.cfg_model.ENCODER_IMG == self.cfg_model.ENCODER_TEXT
            encoder = self.clip.visual
        elif name == 'SwinTransformer':
            encoder = SwinTransformer(
                img_size=384,
                patch_size=4,
                in_chans=3,
                num_classes=21841,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False
            )
            ckpt = torch.load('/data/dyh/models/SwinTransformer/swin_base_patch4_window12_384_22k.pth')
            encoder.load_state_dict(ckpt['model'], strict=False)
        return encoder

    @staticmethod
    def get_text_encoder(cfg_model):
        encoder_name = cfg_model.ENCODER_TEXT
        if 'roberta' in encoder_name:
            encoder = RobertaModel.from_pretrained(encoder_name)
        elif 'bert' in encoder_name:
            encoder = BertModel.from_pretrained(encoder_name, local_files_only=True)
        else:
            return None
        if cfg_model.FREEZE_TEXT_ENCODER:
            for p in encoder.parameters():
                p.requires_grad = False
        return encoder

    def encode_img(self, x, name='crop'):
        crop = x[name]
        if name == 'crop':
            encoder_img = self.encoder_img
            fc_img = self.fc_img
        elif name == 'crop2':
            encoder_img = self.encoder_img2
            fc_img = self.fc_img2
        elif name == 'fg_motionmap':
            encoder_img = self.encoder_fgmotion
            fc_img = self.fc_fgmotion
        if isinstance(crop, list):
            features_crops = list()
            for x_ in crop:
                if 'CLIP' in self.cfg_model.ENCODER_IMG:
                    crop = crop.to(torch.half)
                    features_crop = encoder_img(x_)
                    features_crop = features_crop.to(torch.float)
                    features_crop = features_crop.unsqueeze(2).unsqueeze(3)
                else:
                    features_crop = encoder_img(x_)
                features_crop = fc_img(features_crop).squeeze()
                features_crop = F.normalize(features_crop, p=2, dim=-1)
                features_crops.append(features_crop)
            features_crop = torch.mean(
                torch.stack(
                    features_crops,
                    dim=0
                ),
                dim=0
            )
        else:
            if 'CLIP' in self.cfg_model.ENCODER_IMG:
                crop = crop.to(torch.half)
                features_crop = encoder_img(crop)
                features_crop = features_crop.to(torch.float)
                features_crop = features_crop.unsqueeze(2).unsqueeze(3)
            else:
                features_crop = encoder_img(crop)
            features_crop = fc_img(features_crop).squeeze()
            features_crop = F.normalize(features_crop, p=2, dim=-1)
        if self.cfg_model.USE_MOTION:
            motionmap = x['motionmap']
            features_motion = self.encoder_motion(motionmap)
            features_motion = self.fc_motion(features_motion).squeeze()
            features_motion = F.normalize(features_motion)
            features_merge = self.fc_merge(torch.cat([features_crop, features_motion], dim=-1))
            features_merge = F.normalize(features_merge)
            return features_merge
        else:
            return features_crop

    def encode_text(self, x, name):
        if 'CLIP' in self.cfg_model.ENCODER_TEXT:
            features_text = self.clip.encode_text(x[name])
            features_text = features_text.to(torch.float)
        else:
            features_text = self.encoder_text(
                x['{}_input_ids'.format(name)],
                attention_mask=x['{}_attention_mask'.format(name)]
            )
            features_text = torch.mean(features_text.last_hidden_state, dim=1)
        features_text = getattr(self, 'fc_{}'.format(name))(features_text)
        features_text = F.normalize(features_text, p=2, dim=-1)
        return features_text

    def forward(self, x):
        """
        keys of x: crop, text_input_ids, text_attention_mask,
        """
        features_crops = [
            self.encode_img(x, 'crop')
        ]
        features_texts = [
            self.encode_text(x, 'text')
        ]
        if self.cfg_model.USE_BIGGER_IMAGE:
            features_crops += [
                self.encode_img(x, 'crop2')
            ]
        if self.cfg_model.USE_FgMOTION:
            features_crops += [
                self.encode_img(x, 'fg_motionmap')
            ]
        if self.cfg_model.USE_SINGLE_TEXT:
            features_texts += [
                self.encode_text(x, 'text1'),
                self.encode_text(x, 'text2'),
                self.encode_text(x, 'text3')
            ]
        if self.cfg_model.USE_COLOR_TYPE:
            features_texts += [
                self.encode_text(x, 'color_type')
            ]
        if self.cfg_model.SELF_ATTENTION_WEIGHTS:
            '''图像自注意力特征'''
            f_crops = torch.stack(features_crops, dim=1)  # [B,T,C]
            b, t, c = f_crops.size()
            f_crops = self.attention_img(f_crops) + f_crops  # [B,T,C]
            f_crops = f_crops.permute(0, 2, 1)  # [B,C,T]
            f_crops = F.adaptive_avg_pool1d(f_crops, 1)
            f_crops = f_crops.view(b, c)  # [B,C]
            features_crops.append(f_crops)
            '''文本自注意力特征'''
            f_texts = torch.stack(features_texts, dim=1)
            b, t, c = f_texts.size()
            f_texts = self.attention_text(f_texts) + f_texts
            f_texts = f_texts.permute(0, 2, 1)  # [B,C,T]
            f_texts = F.adaptive_avg_pool1d(f_texts, 1)
            f_texts = f_texts.view(b, c)  # [B,C]
            features_texts.append(f_texts)
        cls_logits_results = []
        if self.cfg_model.INSTANCELOSS:
            cls_logits_results.append(self.fc_share(features_crop))
            cls_logits_results.append(self.fc_share(features_text))
        elif self.cfg_model.ID_LOSS:
            for f_crop in features_crops:
                cls_logits_results.append(self.classifier_img(f_crop))
        return features_crops, features_texts, self.tau, cls_logits_results

if __name__ == '__main__':
    cfg = get_default_config()
    cfg.MODEL.ENCODER_TEXT = 'CLIP_ViT-B/32'
    m = MultiStreamNetwork(cfg.MODEL)