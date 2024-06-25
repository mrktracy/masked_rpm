import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn

class DynamicWeightingRNN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.1,
                 output_dim=2):
        super(DynamicWeightingRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output from the last time step
        out = out[:, -1, :]

        # Pass through the final fully connected layer
        out = self.fc(out)

        # Apply softmax to get the weights
        out = F.softmax(out, dim=-1)

        return out.squeeze(0)

class DynamicWeighting(nn.Module):
    def __init__(self,
                 embed_dim=20,
                 mlp_ratio=2,
                 mlp_drop=0.1,
                 output_dim = 2):
        super(DynamicWeighting, self).__init__()

        self.lin1 = nn.Linear(embed_dim, embed_dim*mlp_ratio)
        self.drop1 = nn.Dropout(p=mlp_drop)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(embed_dim*mlp_ratio, embed_dim*mlp_ratio)
        self.drop2 = nn.Dropout(p=mlp_drop)
        self.lin3 = nn.Linear(embed_dim*mlp_ratio, embed_dim)
        self.drop3 = nn.Dropout(p=mlp_drop)
        self.lin4 = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.drop1(self.lin1(x)))
        x = self.relu(self.drop2(self.lin2(x)))
        x = self.relu(self.drop3(self.lin3(x)))
        x = F.softmax(self.lin4(x), dim=-1)

        return x.squeeze(0)

class TransformerModelv23_ST(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=256,
                 symbol_factor = 1,
                 grid_size = 3,
                 bb_depth = 1,
                 bb_num_heads = 2,
                 use_hadamard = False,
                 per_mlp_drop=0.3):

        super(TransformerModelv23_ST, self).__init__()

        self.embed_dim = embed_dim
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads
        self.use_hadamard = use_hadamard

        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.bb_depth, num_heads=bb_num_heads,
                                             mlp_drop=per_mlp_drop)

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim, mlp_drop=per_mlp_drop)

    def forward(self, sentences):
        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # sentences is (B, 8, 9, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped) # embed_reshaped is (B*9*8, embed_dim)

        x = embed_reshaped.view(batch_size, 8, 9, self.embed_dim)

        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation, embeddings

    def encode(self, images):
        embeddings = self.perception.forward(images) # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings) # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images

class TransformerModelv23(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=256,
                 patch_size = 10,
                 symbol_factor = 1,
                 grid_size = 3,
                 trans_num_heads=4,
                 abs_1_num_heads=4,
                 abs_2_num_heads=4,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 trans_depth = 2,
                 abs_1_depth = 2,
                 abs_2_depth = 2,
                 bb_depth = 1,
                 bb_num_heads = 2,
                 use_hadamard = False,
                 mlp_drop = 0.5,
                 proj_drop = 0.5,
                 attn_drop = 0.5,
                 per_mlp_drop=0.3,
                 topk=5):

        super(TransformerModelv23, self).__init__()

        assert abs_1_depth >= 2, 'Abstractor 1 depth must be at least 2'
        assert abs_2_depth >= 2, 'Abstractor 2 depth must be at least 2'

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads
        self.use_hadamard = use_hadamard
        self.topk = topk

        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.bb_depth, num_heads=bb_num_heads,
                                             mlp_drop=per_mlp_drop)

        self.model_dim = 2*self.embed_dim

        self.tcn_1 = TemporalContextNorm(num_features=self.model_dim)
        self.tcn_2 = TemporalContextNorm(num_features=self.model_dim)

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks_abs_1 = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_1_num_heads,\
                  mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=proj_drop, \
                  attn_drop=attn_drop, drop_path=0.5*((i+1)/abs_1_depth))
            for i in range(abs_1_depth)])

        self.blocks_abs_2 = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_2_num_heads, \
                  mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=proj_drop, \
                  attn_drop=attn_drop, drop_path=0.5 * ((i + 1) / abs_2_depth))
            for i in range(abs_2_depth)])

        self.blocks_trans = nn.ModuleList([
            Block(self.model_dim, self.model_dim, trans_num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=proj_drop, \
                  attn_drop=attn_drop, drop_path=0.5 * ((i + 1) / trans_depth))
            for i in range(trans_depth)])

        self.norm_x_1 = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_x_2 = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_y = norm_layer(self.model_dim)

        self.mlp1 = nn.Linear(self.model_dim + 2 * self.model_dim * self.symbol_factor, self.embed_dim)

        self.relu = nn.ReLU()

        self.mlp2 = nn.Linear(self.embed_dim, 1)

        self.dropout = nn.Dropout(p=mlp_drop)

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim, mlp_drop=per_mlp_drop)

        # define symbols
        normal_initializer = torch.nn.init.normal_
        self.symbols_1 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))
        self.symbols_2 = nn.Parameter(normal_initializer(torch.empty(6, self.model_dim * self.symbol_factor)))

    def ternary_operation(self, x):
        """
        Perform the ternary operation C(x1, x2, x3) for rows and columns across the sequence.
        Input x is of shape (batch_size, 9, embed_dim)
        """

        slice_idx = torch.tensor([0, 3, 6, 0, 1, 2])
        increment = torch.tensor([1, 1, 1, 3, 3, 3])

        # Extract x1, x2, x3 for all sliding windows
        x1 = x[:, slice_idx, :].unsqueeze(3)  # Shape: (batch_size, 6, embed_dim, 1)
        x2 = x[:, slice_idx + increment, :].unsqueeze(2)  # Shape: (batch_size, 6, 1, embed_dim)
        x3 = x[:, slice_idx + 2 * increment, :].unsqueeze(3)  # Shape: (batch_size, 6, embed_dim, 1)

        # Compute the outer product
        outer_product = torch.matmul(x1, x2)  # Shape: (batch_size, 6, embed_dim, embed_dim)

        # Matrix-vector multiplication on the last two dimensions
        result = torch.matmul(outer_product, x3)

        # Squeeze to remove singleton dimension
        result = result.squeeze(-1)  # Shape: (batch_size, 6, embed_dim)

        return result

    def ternary_hadamard(self, x):
        """
        Perform the Hadamard product operation for rows and columns in the sequence.
        Input x is of shape (batch_size, 9, embed_dim).
        """

        slice_idx = torch.tensor([0, 3, 6, 0, 1, 2])
        increment = torch.tensor([1, 1, 1, 3, 3, 3])

        # Extract x1, x2, x3 for all sliding windows
        x1 = x[:, slice_idx, :].unsqueeze(3)  # Shape: (batch_size, 6, embed_dim, 1)
        x2 = x[:, slice_idx + increment, :].unsqueeze(2)  # Shape: (batch_size, 6, 1, embed_dim)
        x3 = x[:, slice_idx + 2 * increment, :].unsqueeze(3)  # Shape: (batch_size, 6, embed_dim, 1)

        # element-wise multiplication
        result = x1 * x2 * x3

        return result

    def forward(self, sentences):
        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # sentences is (B, 8, 9, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped) # x_reshaped is (B*9*8, self.patch_size**2, 256)

        # reshape for concatenating positional embeddings
        x_1 = embed_reshaped.view(batch_size * self.patch_size**2, 8, 9, -1) # x is (B * self.patch_size**2, 8, 9, 256)
        embeddings = x_1.clone()

        # expand positional embeddings to fit batch (B * self.patch_size**2, 8, 9, embed_dim)
        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size * self.patch_size**2, 8, 9, -1)

        # concatenate positional embeddings
        x_1 = torch.cat([x_1, final_pos_embed], dim=-1)

        x_1_reshaped = x_1.view(batch_size * self.patch_size**2 * 8, 9, self.model_dim)

        x_ternary = self.ternary_hadamard(x_1_reshaped) if self.use_hadamard else self.ternary_operation(x_1_reshaped)
        x_2 = x_ternary.view(batch_size * self.patch_size**2, 8, 6, -1)

        # apply temporal context normalization
        x_1 = self.tcn_1(x_1)
        x_2 = self.tcn_2(x_2)

        # reshape x for batch processing
        x_1 = x_1.view(batch_size * self.patch_size**2 * 8, 9, -1)
        x_2 = x_2.view(batch_size * self.patch_size**2 * 8, 6, -1)

        # clone x for passing to transformer blocks
        y = x_1.clone()
        y_skip = x_1.clone() # create copies for skip connection


        # selector = torch.cat((torch.ones(1, 1, self.embed_dim), \
        #                       torch.zeros(1, 1, self.embed_dim)), dim = -1).to(y.device)
        # y_pos = y*selector # broadcasting will take care of dimensions

        # repeat symbols along ,batch dimension
        symbols_1 = self.symbols_1.unsqueeze(0)
        symbols_1 = symbols_1.repeat(batch_size * self.patch_size**2 * 8, 1, 1)
        symbols_2 = self.symbols_2.unsqueeze(0)
        symbols_2 = symbols_2.repeat(batch_size * self.patch_size**2 * 8, 1, 1)

        # multi-headed self-attention blocks of abstractor
        for idx, blk in enumerate(self.blocks_abs_1):
            if idx == 0:
                x_1 = blk(x_q=x_1, x_k=x_1, x_v=symbols_1)
            else:
               x_1 = blk(x_q=x_1, x_k=x_1, x_v=x_1)

        x_1 = self.norm_x_1(x_1)

        # multi-headed self-attention blocks of abstractor
        for idx, blk in enumerate(self.blocks_abs_2):
            if idx == 0:
                x_2 = blk(x_q=x_2, x_k=x_2, x_v=symbols_2)
            else:
                x_2 = blk(x_q=x_2, x_k=x_2, x_v=x_2)

        x_2 = self.norm_x_2(x_2)

        # multi-headed self-attention blocks of transformer
        for idx, blk in enumerate(self.blocks_trans):
            y = blk(x_q=y, x_k=y, x_v=y)

        y = self.norm_y(y)
        y = y + y_skip # add skip connection

        x_1 = x_1.view([batch_size * self.patch_size**2, 8, 9, -1])

        x_2 = x_2.view([batch_size * self.patch_size**2, 8, 6, -1])

        y = y.view(batch_size * self.patch_size**2, 8, 9, -1)
        y = self.tcn_1.inverse(y)

        z = torch.cat([x_1, y], dim=-1)

        z_reshaped = torch.cat([z.mean(dim=-2), x_2.mean(dim=-2)], dim=-1).view(batch_size * self.patch_size**2 * 8, -1)
        z_reshaped = self.mlp1(z_reshaped)
        z_reshaped = self.dropout(z_reshaped)
        dist_reshaped = self.mlp2(self.relu(z_reshaped)) # dist_reshaped is (B * self.patch_size**2 * 8, 1)

        dist, _ = torch.topk(dist_reshaped.view(batch_size, self.patch_size**2, 8), k = self.topk, dim = -2)
        dist = dist.mean(dim=-2)

        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation, embeddings

    def encode(self, images):
        embeddings = self.perception.forward(images) # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings) # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images

class TemporalContextNorm(nn.Module):
    def __init__(self, num_features=768, eps=1e-5, affine=True):
        super(TemporalContextNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features))
            self.beta = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)

    def forward(self, x):
        # x has shape (batch_size, 8, 9, num_features)
        self.mean = x.mean(dim=-2, keepdim=True)
        self.var = x.var(dim=-2, unbiased=False, keepdim=True)

        # Normalize
        x_norm = (x - self.mean) / (self.var + self.eps).sqrt()

        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def inverse(self, x_norm):

        # Invert normalization
        if self.affine:
            x = (x_norm - self.beta) / self.gamma
        else:
            x = x_norm

        x = x * (self.var + self.eps).sqrt() + self.mean

        return x


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

class ResNetDecoder(nn.Module):
    def __init__(self, embed_dim=256, mlp_drop=0.5):
        super(ResNetDecoder, self).__init__()

        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # N, 256, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 128, 20, 20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 64, 40, 40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 32, 80, 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # N, 16, 160, 160
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1), # N, 1, 160, 160
            nn.Sigmoid()  # to ensure the output is in [0, 1] as image pixel intensities
        )

    def forward(self, x):
        x = x.view(x.size(0), 512, 5, 5)
        return self.decoder(x)

class BackbonePerception(nn.Module):
    def __init__(self, embed_dim, num_heads=32, mlp_ratio=4, norm_layer=nn.LayerNorm, depth=4, mlp_drop=0.3):
        super(BackbonePerception, self).__init__()

        self.embed_dim = embed_dim

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.encoder = nn.Sequential( # from N, 1, 160, 160
            ResidualBlock(1, 16), # N, 16, 160, 160
            ResidualBlock(16, 32, 2), # N, 32, 80, 80
            ResidualBlock(32, 64, 2), # N, 64, 40, 40
            ResidualBlock(64, 128, 2), # N, 128, 20, 20
            ResidualBlock(128, 256, 2), # N, 256, 10, 10
            ResidualBlock(256, 512, 2)  # N, 512, 5, 5
        )

        self.blocks = nn.ModuleList([
            Block(512, 512, self.num_heads, self.mlp_ratio, \
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5*((i+1)/self.depth)) for i in range(self.depth)])

        # self.mlp = nn.Linear(256*10*10, self.embed_dim)
        # self.dropout = nn.Dropout(p=mlp_drop)

    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, 512, 5*5)
        x = x.transpose(1,2)

        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        x = x.reshape(batch_dim, 5*5, 512)

        return x

""" Modification of "Vision Transformer (ViT) in PyTorch"
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
A PyTorch implement of Vision Transformers as described in:
Hacked together by / Copyright 2020, Ross Wightman
"""

class Attention(nn.Module):

    def __init__(
            self,
            dim_kq,
            dim_v,
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

        assert dim_kq % num_heads == 0, 'dim_kq should be divisible by num_heads'
        assert dim_v % num_heads == 0, 'dim_v should be divisible by num_heads'

        self.dim_kq = dim_kq
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim_kq = dim_kq // num_heads
        self.head_dim_v = dim_v // num_heads
        self.scale = self.head_dim_kq ** -0.5

        self.w_qs = nn.Linear(dim_kq, dim_kq, bias=q_bias)
        self.w_ks = nn.Linear(dim_kq, dim_kq, bias=k_bias)
        self.w_vs = nn.Linear(dim_v, dim_v, bias=v_bias)

        self.q_norm = norm_layer(self.head_dim_kq) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim_kq) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_v, dim_v)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v):

        batch_size, len_q, len_k, len_v, c = x_q.size(0), x_q.size(1), x_k.size(1), x_v.size(1), self.dim_v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x_q).view(batch_size, self.num_heads, len_q, self.head_dim_kq)
        k = self.w_ks(x_k).view(batch_size, self.num_heads, len_k, self.head_dim_kq)
        v = self.w_vs(x_v).view(batch_size, self.num_heads, len_v, self.head_dim_v)

        q, k = self.q_norm(q), self.k_norm(k)

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
            dim_kq,
            dim_v,
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
            mlp_layer=Mlp
    ):
        super().__init__()
        self.norm1 = norm_layer(dim_kq)
        self.norm1_v = norm_layer(dim_v)
        self.norm2 = norm_layer(dim_v)
        self.norm3 = norm_layer(dim_v)
        self.attn = Attention(
            dim_kq,
            dim_v,
            num_heads=num_heads,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.ls1 = LayerScale(dim_kq, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = mlp_layer(
            in_features=dim_v,
            hidden_features=int(dim_v * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop
        )
        self.ls2 = LayerScale(dim_v, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_k, x_v, use_mlp_layer=True):

        x = x_v + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1_v(x_v))))

        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        else:
            x = self.norm2(x)

        return x