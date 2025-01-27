import math
from functools import partial

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List

from basicsr.archs.vqgan_arch import *
from basicsr.archs.vqvae_arch import VQVAE
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    b, c = size[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model - MLP
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        # self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2*in_ch, out_ch)

        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2 * C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


# @ARCH_REGISTRY.register()
class CodeFormer2(VQVAE):
    def __init__(self, dim_embd=640, n_head=8, n_layers=9,
                codebook_size=4096, latent_size=32,
                 connect_list=['32', '64', '128', '256'], vqvae_path=None):
        super(CodeFormer2, self).__init__(
            vocab_size=codebook_size, z_channels=32, ch=160, test_mode=False,
            share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        )
        if vqvae_path is not None:
            self.load_state_dict(torch.load(vqvae_path, map_location='cpu'))

        for param in self.parameters():
            param.requires_grad = False
        # training_modules = ['encoder', 'quant_conv']
        fix_modules = ['decoder', 'quantize', 'post_quant_conv']
        for module in fix_modules:
            for param in getattr(self, module).parameters():
                param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        self.position_emb = nn.Parameter(torch.zeros(680, self.dim_embd))
        # self.feat_emb = nn.Linear(latent_size * len(self.vae.v_patch_nums), self.dim_embd)
        self.feat_emb = nn.ModuleList(
            nn.Embedding(codebook_size, self.dim_embd)
            for (ph, pw) in self.patch_hws
        )

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ################### Encoder #####################
        x = self.quant_conv(self.encoder(x))
        lq_feat = x  # B 32 16 16
        idx_list = self.quantize.f_to_idxBl_or_fhat(lq_feat, to_fhat=False)
        pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)

        idx_embed_all = []
        for i, idx in enumerate(idx_list):
            idx_embed_all.append(self.feat_emb[i](idx))
        idx_embed = torch.cat(idx_embed_all, dim=1).permute(1, 0, 2)
        # torch.Size([256, 2, 640])
        query_emb = idx_embed
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # torch.Size([256, 2, 640])
        logits = self.idx_pred_layer(query_emb)
        logits = logits.permute(1,0,2)

        return logits, lq_feat

if __name__ == '__main__':
    code_former = CodeFormer2(
        dim_embd=512,
        n_head=8,
        n_layers=9,
        latent_size=32,
        codebook_size=4096,
        connect_list=['32', '64', '128', '256'],
        vqvae_path=r'/Users/katz/Downloads/vae_ch160v4096z32.pth',
    )
    inp = torch.randn(2, 3, 256, 256)
    logits, lq_feat = code_former(inp, w=0.1, code_only=True)
    print(logits.shape)
    print(lq_feat.shape)

    # =====
    # ckpt = '../../net_g_14000.pth'
    # state_dict = torch.load(ckpt, map_location='cpu')
    # print(state_dict['params_ema'].keys())
    # code_former.load_state_dict(state_dict['params_ema'])
    #
    # import glob
    # import math
    #
    # import PIL.Image as PImage
    # from torchvision.transforms import InterpolationMode, transforms
    # import torch
    #
    #
    # def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    #     return x.add(x).add_(-1)
    #
    #
    # def img_folder_to_tensor(img_folder: str, transform: transforms.Compose) -> torch.Tensor:
    #     img_list = glob.glob(f'{img_folder}/*.png')
    #     img_all = []
    #     for img_path in img_list:
    #         img_tensor = transform(PImage.open(img_path))
    #         img_all.append(img_tensor)
    #     img_tensor = torch.stack(img_all, dim=0)
    #     return img_tensor
    #
    #
    # def tensor_to_img(img_tensor: torch.Tensor) -> PImage.Image:
    #     B, C, H, W = img_tensor.shape
    #     assert int(math.sqrt(B)) * int(math.sqrt(B)) == B
    #     b = int(math.sqrt(B))
    #     img_tensor = torch.permute(img_tensor, (1, 0, 2, 3))
    #     img_tensor = torch.reshape(img_tensor, (C, b, b * H, W))
    #     img_tensor = torch.permute(img_tensor, (0, 2, 1, 3))
    #     img_tensor = torch.reshape(img_tensor, (C, b * H, b * W))
    #     img = transforms.ToPILImage()(img_tensor)
    #     return img
    #
    #
    # vae_ckpt = r'/Users/katz/Downloads/vae_ch160v4096z32.pth'
    # B, C, H, W = 4, 3, 256, 256
    # vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=False,
    #             share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).to('cpu')
    # vae.eval()
    # vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    #
    # mid_reso = 1.125
    # final_reso = 256
    # mid_reso = round(min(mid_reso, 2) * final_reso)
    # aug = transforms.Compose(
    #     [
    #         # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
    #         # transforms.CenterCrop((final_reso, final_reso)),
    #         transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
    #         transforms.ToTensor()
    #     ]
    # )
    # inp = img_folder_to_tensor('../../tmp', aug)
    # print(inp.shape)
    #
    # img = tensor_to_img(inp)
    # img.save('../../inp.png')
    # # inp = torch.randn(2, 3, 256, 256)
    #
    # logits = code_former(inp, w=0.1, code_only=True)
    #
    # print(logits.shape)  # torch.Size([2, 680, 640])
    # pred = torch.argmax(logits, dim=-1)
    # splits = [ph*pw for (ph, pw) in code_former.vae.patch_hws]
    # pred = list(torch.split(pred, splits, dim=1))
    # for p in pred:
    #     print('p', p.shape)
    # res_img = code_former.vae.idxBl_to_img(pred, same_shape=True)
    # for i, res in enumerate(res_img):
    #     print('res', res.shape)
    #     res_i = tensor_to_img(res)
    #     res_i.save(f'../../out_{i}.png')
