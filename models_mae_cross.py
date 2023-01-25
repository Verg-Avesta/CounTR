import time
from functools import partial
import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

from timm.models.vision_transformer import PatchEmbed, Block
from models_crossvit import CrossAttentionBlock

from util.pos_embed import get_2d_sincos_pos_embed

class SupervisedMAE(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches , embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.shot_token = nn.Parameter(torch.zeros(512))

        # Exemplar encoder with CNN
        self.decoder_proj1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[3,64,64]->[64,32,32]
        )
        self.decoder_proj2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) #[64,32,32]->[128,16,16]
        )
        self.decoder_proj3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # [128,16,16]->[256,8,8]
        )
        self.decoder_proj4 = nn.Sequential(
            nn.Conv2d(256, decoder_embed_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
            # [256,8,8]->[512,1,1]
        )


        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # Density map regresssion module
        self.decode_head0 = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True)
        )
        self.decode_head3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1)
        )  
    
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.shot_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, y_, shot_num=3):
        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed
        x = x + self.decoder_pos_embed

        # Exemplar encoder
        y_ = y_.transpose(0,1) # y_ [N,3,3,64,64]->[3,N,3,64,64]
        y1=[]
        C=0
        N=0
        cnt = 0
        for yi in y_:
            cnt+=1
            if cnt > shot_num:
                break
            yi = self.decoder_proj1(yi)
            yi = self.decoder_proj2(yi)
            yi = self.decoder_proj3(yi)
            yi = self.decoder_proj4(yi)
            N, C,_,_ = yi.shape
            y1.append(yi.squeeze(-1).squeeze(-1)) # yi [N,C,1,1]->[N,C]       
            
        if shot_num > 0:
            y = torch.cat(y1,dim=0).reshape(shot_num,N,C).to(x.device)
        else:
            y = self.shot_token.repeat(y_.shape[1],1).unsqueeze(0).to(x.device)
        y = y.transpose(0,1) # y [3,N,C]->[N,3,C]
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, y)
        x = self.decoder_norm(x)
        
        # Density map regression
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)

        x = F.interpolate(
                        self.decode_head0(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head1(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head2(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = F.interpolate(
                        self.decode_head3(x), size=x.shape[-1]*2, mode='bilinear', align_corners=False)
        x = x.squeeze(-3)

        return x

    def forward(self, imgs, boxes, shot_num):
        # if boxes.nelement() > 0:
        #     torchvision.utils.save_image(boxes[0], f"data/out/crops/box_{time.time()}_{random.randint(0, 99999):>5}.png")
        with torch.no_grad():
            latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, boxes, shot_num)  # [N, 384, 384]
        return pred


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = SupervisedMAE(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim4(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_fim6(**kwargs):
    model = SupervisedMAE(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  
mae_vit_base4_patch16 = mae_vit_base_patch16_fim4 # decoder: 4 blocks
mae_vit_base6_patch16 = mae_vit_base_patch16_fim6 # decoder: 6 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  
