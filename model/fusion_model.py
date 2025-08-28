import torch
import torch.nn as nn

# 下采样 分辨率2倍，通道4倍
class Downsample(nn.Module):
    def __init__(self, n_feat, out_channels):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),  # 降采样
            nn.Conv2d(in_channels=n_feat * 4, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=False)
        )


    def forward(self, x):
        return self.body(x)

# 上采样 分辨率2， 通道4
class Upsample(nn.Module):
    def __init__(self, n_feat,out_channels):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(in_channels=n_feat//4, out_channels=out_channels, kernel_size=3, stride=1,
                                            padding=1,
                                            bias=False)
                                  )

    def forward(self, x):
        return self.body(x)



import torch.nn.functional as F
class concat(nn.Module):
    def __init__(self):
        super(concat, self).__init__()

    def forward(self, x):
        return torch.cat([x[0],x[1]],dim=1)

from timm.models.layers import DropPath, to_2tuple
import numpy as np
class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        #self.init_weights()


    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x

class TransFusion(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x = self.drop(self.pos_emb + token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return torch.cat([rgb_fea_out, ir_fea_out],dim=1)


class CNNFusion(nn.Module):
    def __init__(self,channel, reduction=16):
        super().__init__()
        self.channel = channel
        self.mask_map_r = nn.Conv2d(channel, 1,1,1,0,bias=True)
        self.mask_map_i = nn.Conv2d(channel, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(channel, channel, 3,1,1,bias=False)
        self.bottleneck2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=False)


    def forward(self, x):
        x_left_ori, x_right_ori = x[0], x[1]
        x_left, x_right = x_left_ori*0.5, x_right_ori*0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)
        out = (torch.cat([out_RGB, out_IR], 1))
        return out

class ENF(nn.Module):
    # 异构动态融合部分的创新
    def __init__(self,in_channels):
        super(ENF, self).__init__()
        dim = in_channels
        self.dim = dim
        self.num_experts =9 # 这个
        self.top_k = 1

        # 定义多个专家，每个专家是 (B, C, W, H) -> (B, C, W, H)
        self.experts = nn.ModuleList([
            TransFusion(d_model=dim), # 这个
            CNNFusion(channel=dim),
            concat(),
            TransFusion(d_model=dim),  # 这个
            CNNFusion(channel=dim),
            concat(),
            TransFusion(d_model=dim),  # 这个
            CNNFusion(channel=dim),
            concat(),
        ])
        self.conv = nn.Conv2d(dim*2, dim, 3,1,1)

        # 门控网络（根据 text_feature 选择专家）
        self.gating_network = nn.Linear(dim, self.num_experts)

    def forward(self, x):


        xf = torch.abs(x[0]-x[1])
        B, C, H, W = xf.shape  # 保持 (B, C, H, W) 格式
        #print(self.dim, xf.shape)
        pool_x = F.adaptive_avg_pool2d(xf, (1, 1)).squeeze(2).squeeze(2)
        #print(pool_x.shape)

        # 计算 gating 权重 (B, num_experts)，根据 text_feature 计算专家选择概率
        #print(self.dim, pool_x.shape)
        gate_logits = self.gating_network(pool_x)  # (B, num_experts)

        gate_weights = F.softmax(gate_logits, dim=-1)  # (B, num_experts)

        # 选择 top-k 专家
        topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)  # (B, top_k)

        # 初始化 MoE 输出
        moe_output = torch.zeros_like(torch.cat([x[0],x[1]], dim=1))  # (B, C, H, W)
        #print(topk_indices.shape)

        # 仅计算 Top-k 专家
        #print(self.top_k)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # (B,)
            weight = topk_values[:, i].view(B, 1, 1, 1)  # (B, 1, 1, 1) 用于加权

            # 计算当前专家输出，仅在选中的 batch 进行计算
            #print(self.num_experts)
            for j in range(self.num_experts):
                mask = (expert_idx == j).view(B, 1, 1, 1)  # 选中的 batch
                if mask.any():
                    #print(moe_output.shape, self.experts[j](x).shape)
                    moe_output += weight * mask * self.experts[j](x)  # (B, C, H, W)

        return self.conv(moe_output)
        #print(self.experts[0](x).shape, self.experts[1](x).shape)
        #return self.experts[2](x)