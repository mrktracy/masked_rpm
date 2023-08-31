import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn

class TransformerModelMNIST(nn.Module):
    def __init__(self, embed_dim=512, num_heads=16, mlp_ratio=4., \
                 norm_layer=nn.LayerNorm, con_depth=8, can_depth=8, guess_depth=8, cat=True):
        super(TransformerModelMNIST, self).__init__()

        self.model_dim = embed_dim

        self.perception = ResNetEncoder(embed_dim=self.model_dim)

        self.con_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(con_depth)])

        self.can_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(can_depth)])

        self.first_guess_block = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            ])

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth-1)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.lin = nn.Linear(self.model_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        # apply perception module to all images
        x_reshaped = x.view(-1, 1, 160, 160)  # x is (B, 16, 1, 160, 160)
        y_reshaped = self.perception(x_reshaped)
        y = y_reshaped.view(batch_size, 16, -1)

        context = y[:,0:8,:] # x is (B, 16, embed_dim)
        candidates = y[:,8:,:]

        for blk in self.con_blocks: # multi-headed self-attention layer
            context = blk(x_q=context, x_k=context, x_v=context)

        for blk in self.can_blocks:
            candidates = blk(x_q=candidates, x_k=candidates, x_v=candidates)

        # for blk1,blk2 in self.first_guess_block:
        #     z = blk1(x_q=context, x_k=candidates, x_v=candidates)
        #     z = blk2(x_q=candidates, x_k=context, x_v=z)
        #
        # for blk1, blk2 in self.guess_blocks:
        #     z = blk1(x_q=context, x_k=z, x_v=z)
        #     z = blk2(x_q=candidates, x_k=context, x_v=z)

        for blk1,blk2 in self.first_guess_block:
            z = blk1(x_q=context, x_k=candidates, x_v=candidates)
            z = blk2(x_q=candidates, x_k=z, x_v=z)

        for blk1, blk2 in self.guess_blocks:
            z = blk1(x_q=context, x_k=z, x_v=z)
            z = blk2(x_q=candidates, x_k=z, x_v=z)

        z_reshaped = z.view(-1,self.model_dim)
        guess_reshaped = self.lin(z_reshaped)
        guess = guess_reshaped.view(batch_size,8)

        return guess

class TransformerModelv5(nn.Module):
    def __init__(self, embed_dim=512, grid_size=3, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, \
                 abstr_depth=12, reas_depth=12, cat=True):
        super(TransformerModelv5, self).__init__()

        self.cat = cat

        if self.cat == True:
            self.model_dim = 2*embed_dim
        else:
            self.model_dim = embed_dim

        # # initialize and retrieve positional embeddings
        # self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        # pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.perception = ResNetEncoder(embed_dim=embed_dim)

        self.abstr_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(abstr_depth)])

        self.first_reas_block = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
        ])

        self.reas_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(reas_depth-1)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.lin = nn.Linear(self.model_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        # apply perception module to all images
        x_reshaped = x.view(-1,1,160,160) # x is (B, 16, 1, 160, 160)
        y_reshaped = self.perception(x_reshaped)
        y = y_reshaped.view(batch_size,16,-1)

        context = y[:,0:8,:] # y is (B, 16, embed_dim)
        candidates = y[:,8:,:]

        # if self.cat == True:
        #     context = torch.cat([context, self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)],\
        #                         dim=2)  # add positional embeddings
        #     candidates = torch.cat([candidates, self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)],\
        #                         dim=2)  # add 9th positional embedding to all candidates
        # else:
        #     context = context + self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)  # add positional embeddings
        #     candidates = candidates + self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)  # add 9th positional embedding to all candidates

        y = torch.cat([context,candidates], dim=1)

        for blk in self.abstr_blocks: # multi-headed self-attention layer
            y = blk(x_q=y, x_k=y, x_v=y)

        context_enc = y[:, 0:8, :]  # x is (B, 16, embed_dim)
        candidates_enc = y[:, 8:, :]

        for blk1,blk2 in self.first_reas_block:
            z = blk1(x_q=context_enc, x_k=candidates_enc, x_v=candidates_enc)
            z = blk2(x_q=candidates_enc, x_k=z, x_v=z)

        for blk1, blk2 in self.reas_blocks:
            z = blk1(x_q=context_enc, x_k=z, x_v=z)
            z = blk2(x_q=candidates_enc, x_k=z, x_v=z)

        z_reshaped = z.view(-1, self.model_dim)
        guess_reshaped = self.lin(z_reshaped)
        guess = guess_reshaped.view(batch_size, 8)

        return guess

class TransformerModelv4(nn.Module):
    def __init__(self, embed_dim=512, grid_size=3, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, \
                 abstr_depth=12, reas_depth=12, cat=True):
        super(TransformerModelv4, self).__init__()

        self.cat = cat

        if self.cat == True:
            self.model_dim = 2*embed_dim
        else:
            self.model_dim = embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.abstr_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(abstr_depth)])

        self.first_reas_block = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
        ])

        self.reas_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(reas_depth-1)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(self.model_dim*8, self.model_dim)

        self.lin2 = nn.Linear(self.model_dim, 64)

        self.lin3 = nn.Linear(64, 8)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        context = x[:,0:8,:] # x is (B, 16, embed_dim)
        candidates = x[:,8:,:]

        if self.cat == True:
            context = torch.cat([context, self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)],\
                                dim=2)  # add positional embeddings
            candidates = torch.cat([candidates, self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)],\
                                dim=2)  # add 9th positional embedding to all candidates
        else:
            context = context + self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)  # add positional embeddings
            candidates = candidates + self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)  # add 9th positional embedding to all candidates

        x = torch.cat([context,candidates], dim=1)

        for blk in self.abstr_blocks: # multi-headed self-attention layer
            x = blk(x_q=x, x_k=x, x_v=x)

        context_enc = x[:, 0:8, :]  # x is (B, 16, embed_dim)
        candidates_enc = x[:, 8:, :]

        # for blk1,blk2 in self.first_reas_block:
        #     y = blk1(x_q=candidates_enc, x_k=candidates_enc, x_v=candidates_enc, use_mlp_layer=False)
        #     y = blk2(x_q=y, x_k=context_enc, x_v=context_enc)
        #
        # for blk1, blk2 in self.reas_blocks:
        #     y = blk1(x_q=y, x_k=y, x_v=y, use_mlp_layer=False)
        #     y = blk2(x_q=y, x_k=context_enc, x_v=context_enc)

        # possibility two
        for blk1,blk2 in self.first_reas_block:
            y = blk1(x_q=context_enc, x_k=candidates_enc, x_v=candidates_enc, use_mlp_layer=False)
            y = blk2(x_q=candidates_enc, x_k=context_enc, x_v=y)

        for blk1, blk2 in self.reas_blocks:
            y = blk1(x_q=context_enc, x_k=y, x_v=y, use_mlp_layer=False)
            y = blk2(x_q=candidates_enc, x_k=context_enc, x_v=y)

        y = self.flatten(y)
        y = self.relu(self.lin1(y))
        y = self.relu(self.lin2(y))
        y = self.lin3(y)

        return y

''' ResNet Encoder '''
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ResNetEncoder, self).__init__()

        self.embed_dim = embed_dim

        self.encoder = nn.Sequential( # from N, 1, 160, 160
            ResidualBlock(1, 16), # N, 16, 160, 160
            ResidualBlock(16, 32, 2), # N, 32, 80, 80
            ResidualBlock(32, 64, 2), # N, 64, 40, 40
            ResidualBlock(64, 128, 2), # N, 128, 20, 20
            ResidualBlock(128, 256, 2), # N, 256, 10, 10
            nn.Flatten(), # N, 256*10*10
            nn.Linear(256*10*10, self.embed_dim), # N, embed_dim
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

""" Modification of "Vision Transformer (ViT) in PyTorch"
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
A PyTorch implement of Vision Transformers as described in:
Hacked together by / Copyright 2020, Ross Wightman
"""

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.w_qs = nn.Linear(dim, dim, bias=q_bias)
        self.w_ks = nn.Linear(dim, dim, bias=k_bias)
        self.w_vs = nn.Linear(dim, dim, bias=v_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v):

        batch_size, len_q, len_k, len_v, c = x_q.size(0), x_q.size(1), x_k.size(1), x_v.size(1), self.dim

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x_q).view(batch_size, self.num_heads, len_q, self.head_dim)
        k = self.w_ks(x_k).view(batch_size, self.num_heads, len_k, self.head_dim)
        v = self.w_vs(x_v).view(batch_size, self.num_heads, len_v, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, len_q, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            q_bias=False,
            k_bias=False,
            v_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_k, x_v, use_mlp_layer=True):

        x = x_q + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1(x_v))))
        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        else:
            x = self.norm2(x)

        return x


'''' Previous Transformer Models '''''
class TransformerModelv3(nn.Module):
    def __init__(self, embed_dim=512, grid_size=3, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, con_depth=6,\
                 can_depth=4, guess_depth=4, cat=True):
        super(TransformerModelv3, self).__init__()

        self.cat = cat

        if self.cat == True:
            self.model_dim = 2*embed_dim
        else:
            self.model_dim = embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.con_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(con_depth)])

        self.can_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(can_depth)])

        self.first_guess_block = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            ])

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth-1)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.flatten = nn.Flatten()

        self.lin1 = nn.Linear(self.model_dim*8, self.model_dim)

        self.lin2 = nn.Linear(self.model_dim, 64)

        self.lin3 = nn.Linear(64, 8)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        context = x[:,0:8,:] # x is (B, 16, embed_dim)
        candidates = x[:,8:,:]

        if self.cat == True:
            context = torch.cat([context, self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)],\
                                dim=2)  # add positional embeddings
            candidates = torch.cat([candidates, self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)],\
                                dim=2)  # add 9th positional embedding to all candidates
        else:
            context = context + self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)  # add positional embeddings
            candidates = candidates + self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)  # add 9th positional embedding to all candidates

        for blk in self.con_blocks: # multi-headed self-attention layer
            context = blk(x_q=context, x_k=context, x_v=context)

        for blk in self.can_blocks:
            candidates = blk(x_q=candidates, x_k=candidates, x_v=candidates)

        for blk1,blk2 in self.first_guess_block:
            y = blk1(x_q=context, x_k=candidates, x_v=candidates)
            y = blk2(x_q=candidates, x_k=context, x_v=y)

        for blk1, blk2 in self.guess_blocks:
            y = blk1(x_q=context, x_k=y, x_v=y)
            y = blk2(x_q=candidates, x_k=context, x_v=y)

        y = self.flatten(y)
        y = self.relu(self.lin1(y))
        y = self.relu(self.lin2(y))
        y = self.lin3(y)

        return y

'''' Previous Transformer Model, relying on Block class from timm '''''
class TransformerModelv1(nn.Module):
    def __init__(self, embed_dim=256, grid_size = 3, num_heads=16, mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 4):
        super(TransformerModelv1, self).__init__()

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(embed_dim*2, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])

        self.norm = norm_layer(embed_dim*2)

        # self.flatten = nn.Flatten()
        #
        # self.fc1 = nn.Linear(256*9, 256*7)
        #
        # self.fc2 = nn.Linear(256*7, 256*5)
        #
        # self.fc3 = nn.Linear(256*5, 256*3)
        #
        # self.fc4 = nn.Linear(256*3, 256)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x
        x = torch.cat([x, self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1)], dim=2)  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x)
        x = self.norm(x)
        # x = self.flatten(x) # flatten
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        return x