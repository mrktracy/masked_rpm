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


class SAViRt(nn.Module):
    """
    SAViR-T: Spatially Attentive Visual Reasoning with Transformers
    This model implements the SAViR-T architecture for solving Raven's Progressive Matrices (RPM).
    """

    def __init__(self,
                 embed_dim=512,
                 grid_dim=5,
                 bb_depth=1,
                 bb_num_heads=2,
                 per_mlp_drop=0.3):
        super(SAViRt, self).__init__()

        self.embed_dim = embed_dim
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads
        self.grid_dim = grid_dim

        self.perception = BackbonePerception(embed_dim=self.embed_dim, grid_dim=self.grid_dim, depth=self.bb_depth, num_heads=bb_num_heads,
                                             mlp_drop=per_mlp_drop)

        self.model_dim = 2 * self.embed_dim

        # Define Φ_MLP for relation extraction
        self.phi_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Define Ψ_MLP for shared rule extraction
        self.psi_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, 2 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.Dropout(p=0.5)
        )

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim)

    def extract_relations(self, x):
        """
        Extract relations from rows and columns of the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 9, grid_dim**2, embed_dim)

        Returns:
            tuple: Row and column relations, each of shape (batch_size, 3, grid_dim**2, embed_dim)
        """
        batch_size, _, num_patches, embed_dim = x.shape

        # Group by rows and columns
        rows = x.view(batch_size, 3, 3, num_patches, embed_dim)
        cols = x.view(batch_size, 3, 3, num_patches, embed_dim).transpose(1, 2)

        # Apply Φ_MLP to each row and column
        row_relations = self.phi_mlp(rows.reshape(-1, 3 * embed_dim)).view(batch_size, 3, num_patches, embed_dim)
        col_relations = self.phi_mlp(cols.reshape(-1, 3 * embed_dim)).view(batch_size, 3, num_patches, embed_dim)

        return row_relations, col_relations

    def extract_shared_rule(self, r_i, r_j):
        """
        Extract shared rules between two relation vectors.

        Args:
            r_i, r_j (torch.Tensor): Relation vectors of shape (batch_size, num_patches, embed_dim)

        Returns:
            torch.Tensor: Shared rule of shape (batch_size, num_patches, embed_dim)
        """
        shared_input = torch.cat([r_i, r_j], dim=-1)
        shared_rule = self.psi_mlp(shared_input)
        return shared_rule

    def forward(self, sentences):
        """
        Forward pass of the SAViR-T model.

        Args:
            sentences (torch.Tensor): Input tensor of shape (batch_size, 8, 9, 1, 160, 160)

        Returns:
            tuple: Distribution over choices and reconstructed images
        """
        batch_size = sentences.size(0)
        # Reshape input: (batch_size * 8 * 9, 1, 160, 160)
        sen_reshaped = sentences.view(-1, 1, 160, 160)
        # Extract features: (batch_size * 8 * 9, grid_dim**2, embed_dim)
        embed_reshaped = self.perception.forward(sen_reshaped)

        # Reshape embeddings for relation extraction: (batch_size*8, 9, grid_dim**2, embed_dim)
        x = embed_reshaped.view(batch_size * 8, 9, self.grid_dim ** 2, -1)

        # Extract relations: (batch_size*8, 3, grid_dim**2, embed_dim)
        row_relations, col_relations = self.extract_relations(x)

        # Extract shared rules between first two rows and columns
        r_12 = self.extract_shared_rule(row_relations[:, 0], row_relations[:, 1])
        c_12 = self.extract_shared_rule(col_relations[:, 0], col_relations[:, 1])

        # Combine row and column shared rules: (batch_size*8, grid_dim**2, 2*embed_dim)
        rc_12 = torch.cat([r_12, c_12], dim=-1)

        # Average across patches: (batch_size*8, 2*embed_dim)
        rc_12 = rc_12.mean(dim=1)

        # Extract shared rules involving the third row/column
        r_1a = self.extract_shared_rule(row_relations[:, 0], row_relations[:, 2])
        c_1a = self.extract_shared_rule(col_relations[:, 0], col_relations[:, 2])
        rc_1a = torch.cat([r_1a, c_1a], dim=-1).mean(dim=1)

        r_2a = self.extract_shared_rule(row_relations[:, 1], row_relations[:, 2])
        c_2a = self.extract_shared_rule(col_relations[:, 1], col_relations[:, 2])
        rc_2a = torch.cat([r_2a, c_2a], dim=-1).mean(dim=1)

        # Average the two shared rules: (batch_size*8, 2*embed_dim)
        rc_a = 0.5 * (rc_1a + rc_2a)

        # Compute similarity score: (batch_size*8,)
        scores = F.cosine_similarity(rc_12, rc_a, dim=-1)
        # Reshape to (batch_size, 8)
        dist = scores.reshape(batch_size, 8)

        # Compute recreation for autoencoder loss: (batch_size, 8, 9, 1, 160, 160)
        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation

    def encode(self, images):
        embeddings = self.perception.forward(images) # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings) # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images


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
    def __init__(self, embed_dim=256):
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
    def __init__(self, out_channels, grid_dim, num_heads=32, mlp_ratio=4, norm_layer=nn.LayerNorm,
                 depth=4, mlp_drop=0.3):
        super(BackbonePerception, self).__init__()

        self.out_channels = out_channels

        self.depth = depth
        self.grid_dim = grid_dim
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

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_dim ** 2, self.out_channels]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.out_channels, grid_size=self.grid_dim, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(self.out_channels*2, self.out_channels*2, self.num_heads, self.mlp_ratio, \
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=mlp_drop, attn_drop=0.1, \
                  drop_path=0.5*((i+1)/self.depth)) for i in range(self.depth)])


    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, self.grid_dim**2, self.out_channels)

        # expand positional embeddings to fit batch (B, self.grid_dim**2, embed_dim)
        pos_embed_final = self.pos_embed.unsqueeze(0).expand(batch_dim, self.grid_dim ** 2, self.out_channels)

        # add positional embeddings
        x = x + pos_embed_final

        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        x = x.reshape(batch_dim, self.grid_dim**2, self.out_channels)

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