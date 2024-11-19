import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
# from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn
import logging


class Perception(nn.Module):
    def __init__(self,
                 embed_dim,
                 grid_size=3,
                 num_candidates=8,
                 bb_depth=1,
                 bb_num_heads=2,
                 per_mlp_drop=0):
        super().__init__()
        self.n_nodes = grid_size ** 2
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_candidates = num_candidates

        # Initialize positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.n_nodes, embed_dim]), requires_grad=False)
        pos_embed_data = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float())

        # Backbone for feature extraction
        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=bb_depth, num_heads=bb_num_heads, mlp_drop=per_mlp_drop)

    def forward(self, sentences):
        """
        Args:
            sentences: Tensor of shape (batch_size, num_candidates, grid_size**2, 1, 160, 160)
        """
        batch_size = sentences.size(0)

        # Reshape sentences for processing
        sen_reshaped = sentences.view(-1, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped)  # (batch_size * num_candidates * grid_size**2, embed_dim)

        # Reshape back and add positional embeddings
        embeddings = embed_reshaped.view(batch_size, self.num_candidates, self.n_nodes, -1)
        pos_embed_expanded = self.pos_embed.unsqueeze(0).expand(batch_size, self.num_candidates, self.n_nodes, -1)

        # Concatenate positional embeddings
        embeddings_final = torch.cat([embeddings, pos_embed_expanded], dim=-1)

        return embeddings_final


class AGMBrain(nn.Module):
    def __init__(self,
                 neuron_dim,
                 n_neurons,
                 n_msg_passing_steps,
                 grid_size,
                 num_candidates,
                 device,
                 input_features):
        super().__init__()
        self.neuron_dim = neuron_dim
        self.n_neurons = n_neurons
        self.n_steps = n_msg_passing_steps
        self.grid_size = grid_size
        self.num_candidates = num_candidates
        self.device = device

        # Input transformation to neuron dimensions
        self.input_proj = nn.Linear(input_features, neuron_dim)

        # Trainable parameters for neuron states and edge vectors
        self.neuron_states = nn.Parameter(torch.randn(n_neurons, neuron_dim))  # Shape: (n_neurons, neuron_dim)
        self.edge_vectors = nn.Parameter(torch.randn(n_neurons, n_neurons, neuron_dim))  # Shape: (n_neurons, n_neurons, neuron_dim)

        # Output projections
        self.recreate_proj = nn.Linear(neuron_dim, input_features)
        self.score_proj = nn.Linear(neuron_dim, 1)

    def forward(self, x):
        batch_cand_size = x.size(0)  # batch_size * num_candidates
        embed_dim_doubled = x.size(1) // (self.grid_size ** 2)

        # Transform input to neuron dimension
        x_transformed = self.input_proj(x)  # Shape: (batch_size * num_candidates, neuron_dim)

        # Initialize neuron states
        states = self.neuron_states.unsqueeze(0).expand(batch_cand_size, -1,
                                                        -1)  # (batch_cand_size, n_neurons, neuron_dim)
        states = states.transpose(1, 2)  # Transpose to (batch_cand_size, neuron_dim, n_neurons)
        states = states + x_transformed.unsqueeze(-1)  # Broadcast input to all neurons

        # Create mask to avoid self-loops
        mask = ~torch.eye(self.n_neurons, dtype=torch.bool, device=x.device)  # Shape: (n_neurons, n_neurons)

        # Message passing
        for _ in range(self.n_steps):
            edge_vecs = self.edge_vectors  # Shape: (n_neurons, n_neurons, neuron_dim)

            # Ensure shapes align
            assert states.size(-1) == edge_vecs.size(
                -2), f"Mismatch: states {states.size(-1)} != edge_vecs {edge_vecs.size(-2)}"

            # Einsum operation to compute messages
            transform_matrices = torch.einsum('bdn,ijn->bdij', states,
                                              edge_vecs)  # Shape: (batch_cand_size, neuron_dim, n_neurons, n_neurons)
            messages = torch.einsum('bdij,bdn->bdi', transform_matrices,
                                    states)  # Shape: (batch_cand_size, neuron_dim, n_neurons)

            # Apply mask to prevent self-loops
            messages = messages * mask.unsqueeze(0).float()  # Broadcast mask along batch dimension

            # Aggregate messages and update states
            new_states = messages.sum(dim=-1)  # Aggregate across neurons: (batch_cand_size, neuron_dim)
            states = F.relu(new_states)  # Apply nonlinearity

        output_states = states[:, :, -1]  # Take final state of the last neuron

        # Project to outputs
        recreation = self.recreate_proj(output_states)
        scores = self.score_proj(output_states)

        # Reshape outputs
        batch_size = batch_cand_size // self.num_candidates
        recreation = recreation.view(batch_size, self.num_candidates, self.grid_size ** 2, embed_dim_doubled)
        scores = scores.view(batch_size, self.num_candidates)

        return recreation, scores


class AsymmetricGraphModel(nn.Module):
    def __init__(self,
                 embed_dim,
                 grid_size,
                 num_candidates,
                 n_msg_passing_steps,
                 bb_depth,
                 bb_num_heads,
                 neuron_dim,
                 n_neurons,
                 device=None):
        super(AsymmetricGraphModel, self).__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_candidates = num_candidates
        self.device = device

        # Perception module
        self.perception = Perception(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_candidates=num_candidates,
            bb_depth=bb_depth,
            bb_num_heads=bb_num_heads
        )

        # Reasoning module
        input_features = grid_size ** 2 * embed_dim * 2
        self.reasoning = AGMBrain(
            neuron_dim=neuron_dim,
            n_neurons=n_neurons,
            n_msg_passing_steps=n_msg_passing_steps,
            grid_size=grid_size,
            num_candidates=num_candidates,
            device=device,
            input_features=input_features
        )

    def forward(self, sentences):
        # Step 1: Perception
        embeddings = self.perception.forward(sentences)

        # Step 2: Flatten embeddings for reasoning
        batch_size = embeddings.size(0)
        grid_features = embeddings.view(batch_size * embeddings.size(1), -1)

        # Step 3: Reasoning
        recreation, scores = self.reasoning.forward(grid_features)

        return embeddings, recreation, scores


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


class BackbonePerception(nn.Module):
    def __init__(self,
                 embed_dim,
                 out_channels=512,
                 grid_dim=5,
                 num_heads=32,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm,
                 depth=4,
                 mlp_drop=0.3):
        super(BackbonePerception, self).__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.grid_dim = grid_dim

        self.encoder = nn.Sequential(  # from N, 1, 160, 160
            ResidualBlock(1, 16),  # N, 16, 160, 160
            ResidualBlock(16, 32, 2),  # N, 32, 80, 80
            ResidualBlock(32, 64, 2),  # N, 64, 40, 40
            ResidualBlock(64, 128, 2),  # N, 128, 20, 20
            ResidualBlock(128, 256, 2),  # N, 256, 10, 10
            ResidualBlock(256, 512, 2)  # N, 512, 5, 5
        )

        self.blocks = nn.ModuleList([
            Block(self.out_channels, self.out_channels, self.num_heads, self.mlp_ratio,
                  q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=0.3, attn_drop=0.3,
                  drop_path=0.5 * ((i + 1) / self.depth), restrict_qk=False) for i in range(self.depth)])

        self.mlp = nn.Linear(self.out_channels * self.grid_dim**2, self.embed_dim)
        self.dropout = nn.Dropout(p=mlp_drop)

    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, self.grid_dim ** 2, self.out_channels)

        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        x = x.reshape(batch_dim, self.out_channels * self.grid_dim**2)

        x = self.dropout(self.mlp(x))

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
            restrict_qk=False
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
        self.w_ks = self.w_qs if restrict_qk else nn.Linear(dim_kq, dim_kq, bias=k_bias)
        self.w_vs = nn.Linear(dim_v, dim_v, bias=v_bias)

        self.q_norm = norm_layer(self.head_dim_kq) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim_kq) if qk_norm else nn.Identity()
        self.qk_norm = norm_layer(self.head_dim_kq) if qk_norm and restrict_qk else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_v, dim_v)
        self.proj_drop = nn.Dropout(proj_drop)
        self.restrict_qk = restrict_qk

    def forward(self, x_q, x_k, x_v):

        batch_size, len_q, len_k, len_v, c = x_q.size(0), x_q.size(1), x_k.size(1), x_v.size(1), self.dim_v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(x_q).view(batch_size, self.num_heads, len_q, self.head_dim_kq)
        k = self.w_ks(x_k).view(batch_size, self.num_heads, len_k, self.head_dim_kq)
        v = self.w_vs(x_v).view(batch_size, self.num_heads, len_v, self.head_dim_v)

        if self.restrict_qk:
            q, k = self.qk_norm(q), self.qk_norm(k)
        else:
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
            mlp_layer=Mlp,
            restrict_qk=False
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
            norm_layer=norm_layer,
            restrict_qk=restrict_qk
        )
        self.ls1 = LayerScale(dim_kq, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = mlp_layer(
            in_features=dim_v,
            hidden_features=int(dim_v * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim_v, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_k, x_v, use_mlp_layer=True):

        x = x_v + self.drop_path1(self.ls1(self.attn.forward(self.norm1(x_q), self.norm1(x_k), self.norm1_v(x_v))))

        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        else:
            x = self.norm2(x)

        return x

