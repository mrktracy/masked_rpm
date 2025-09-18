import torch
import pos_embed as pos
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, DropPath
import logging
from typing import Tuple


class Perception(nn.Module):
    def __init__(self,
                 embed_dim,
                 grid_size=3,
                 num_candidates=8,
                 bb_depth=1,
                 bb_num_heads=2,
                 bb_mlp_ratio=4,
                 bb_proj_drop=0.3,
                 bb_attn_drop=0.3,
                 bb_drop_path_max=0.5,
                 bb_mlp_drop=0,
                 decoder_mlp_drop=0.5,
                 use_bb_pos_enc=False):
        super().__init__()
        self.n_nodes = grid_size ** 2
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_candidates = num_candidates

        # Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.n_nodes, embed_dim]), requires_grad=False)
        pos_embed_data = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float())

        # Backbone for feature extraction
        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=bb_depth, num_heads=bb_num_heads,
                                             mlp_ratio=bb_mlp_ratio, mlp_drop=bb_mlp_drop, proj_drop=bb_proj_drop,
                                             attn_drop=bb_attn_drop, drop_path_max=bb_drop_path_max,
                                             use_bb_pos_enc=use_bb_pos_enc)

        # Decoder for reconstructing sentences
        self.decoder = ResNetDecoder(embed_dim=self.embed_dim, mlp_drop=decoder_mlp_drop)

    def forward(self, sentences):
        """
        Args:
            sentences: Tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        Returns:
            embeddings_final: Tensor of shape [batch_size, num_candidates, grid_size**2, embed_dim]
            reconstructed_sentences: Reconstructed tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        """
        batch_size = sentences.size(0)

        # Reshape for processing by BackbonePerception
        sentences_reshaped = sentences.view(-1, 1, 160, 160)  # Shape: [B * N_c * G^2, 1, 160, 160]
        features = self.perception.forward(sentences_reshaped)  # Shape: [B * N_c * G^2, embed_dim]

        # Reshape back to original context
        features_reshaped = features.view(batch_size, self.num_candidates, self.n_nodes,
                                          -1)  # [B, N_c, G^2, embed_dim]

        # Expand and add positional embeddings
        pos_embed_expanded = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_candidates,
                                                                             self.n_nodes, -1)
        embeddings_final = features_reshaped + pos_embed_expanded  # Element-wise addition

        # Reconstruct sentences
        reconstructed_sentences = self.decoder.forward(features)  # Shape: [B * N_c * G^2, 1, 160, 160]
        reconstructed_sentences = reconstructed_sentences.view(batch_size, self.num_candidates, self.n_nodes, 1,
                                                               160, 160)

        return embeddings_final, reconstructed_sentences


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
                 mlp_drop=0.3,
                 proj_drop=0.3,
                 attn_drop=0.3,
                 drop_path_max = 0.5,
                 use_bb_pos_enc=False):
        super(BackbonePerception, self).__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.grid_dim = grid_dim
        self.use_bb_pos_enc = use_bb_pos_enc

        # CLS Token
        # self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))

        self.encoder = nn.Sequential(  # from N, 1, 160, 160
            ResidualBlock(1, 16),  # N, 16, 160, 160
            ResidualBlock(16, 32, 2),  # N, 32, 80, 80
            ResidualBlock(32, 64, 2),  # N, 64, 40, 40
            ResidualBlock(64, 128, 2),  # N, 128, 20, 20
            ResidualBlock(128, 256, 2),  # N, 256, 10, 10
            ResidualBlock(256, 512, 2)  # N, 512, 5, 5
        )

        if use_bb_pos_enc:
            self.pos_embed = nn.Parameter(torch.zeros([grid_dim ** 2, out_channels]), requires_grad=False)
            pos_embed_data = pos.get_2d_sincos_pos_embed(embed_dim=out_channels, grid_size=grid_dim, cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float())

        self.blocks = nn.ModuleList([
            Block(self.out_channels, self.out_channels, self.num_heads, self.mlp_ratio,
                  q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer,
                  proj_drop=proj_drop, attn_drop=attn_drop,
                  drop_path=drop_path_max * ((i + 1) / self.depth), restrict_qk=False)
            for i in range(self.depth)])

        self.mlp = nn.Linear(self.out_channels * grid_dim ** 2, self.embed_dim)
        self.dropout = nn.Dropout(p=mlp_drop)

    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, self.grid_dim ** 2, self.out_channels)

        # cls_tokens = self.cls_token.expand(batch_dim, -1, -1)
        # x = torch.cat([cls_tokens, x], dim=1)  # CLS token at the start

        if self.use_bb_pos_enc:
            pos_embed = self.pos_embed.unsqueeze(0).expand(batch_dim, -1, self.out_channels)
            x = x + pos_embed

        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        # x = x[:, 0, :]

        x = self.dropout(self.mlp(x.view(batch_dim, -1)))

        return x


class ResNetDecoder(nn.Module):
    def __init__(self, embed_dim=512, mlp_drop=0.5):
        super(ResNetDecoder, self).__init__()
        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 256 * 10 * 10),
            nn.Dropout(p=mlp_drop),
            nn.Unflatten(1, (256, 10, 10)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # N, 128, 20, 20
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # N, 64, 40, 40
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # N, 32, 80, 80
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # N, 16, 160, 160
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # N, 1, 160, 160
            nn.Sigmoid()  # to ensure the output is in [0, 1] as image pixel intensities
        )

    def forward(self, x):
        return self.decoder(x)


class ReasoningModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        grid_size: int,
        abs_depth: int,
        trans_depth: int,
        ternary_depth: int,
        abs_num_heads: int,
        trans_num_heads: int,
        tern_num_heads: int,
        num_candidates: int = 8,
        abs_mlp_ratio: float = 4.0,
        trans_mlp_ratio: float = 4.0,
        tern_mlp_ratio: float = 4.0,
        phi_mlp_hidden_dim: int = 4,
        abs_proj_drop: float = 0.0,
        trans_proj_drop: float = 0.0,
        tern_proj_drop: float = 0.0,
        abs_attn_drop: float = 0.0,
        trans_attn_drop: float = 0.0,
        tern_attn_drop: float = 0.0,
        abs_drop_path_max: float = 0.0,
        trans_drop_path_max: float = 0.0,
        tern_drop_path_max: float = 0.0,
        num_symbols_abs: int = 9,
        num_symbols_ternary: int = 6,
        norm_layer=nn.LayerNorm,
        bb_depth: int = 2,
        bb_num_heads: int = 8,
        bb_mlp_ratio: int = 4,
        bb_proj_drop=0.3,
        bb_attn_drop=0.3,
        bb_drop_path_max=0.5,
        bb_mlp_drop=0,
        decoder_mlp_drop=0.5,
        use_bb_pos_enc=False,
        symbol_factor_abs=1,
        symbol_factor_tern=1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_candidates = num_candidates
        self.symbol_factor_abs = symbol_factor_abs
        self.symbol_factor_tern = symbol_factor_tern
        self.num_symbols_ternary = num_symbols_ternary

        # Initialize Perception module
        self.perception = Perception(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_candidates=8,  # Default assumed; adapt as needed
            bb_depth=bb_depth,
            bb_num_heads=bb_num_heads,
            bb_mlp_ratio=bb_mlp_ratio,
            bb_proj_drop=bb_proj_drop,
            bb_attn_drop=bb_attn_drop,
            bb_drop_path_max=bb_drop_path_max,
            bb_mlp_drop=bb_mlp_drop,
            decoder_mlp_drop=decoder_mlp_drop,
            use_bb_pos_enc=use_bb_pos_enc
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed_data = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float())

        # Temporal normalization
        self.temporal_norm = TemporalNorm(embed_dim)
        self.temporal_norm_tern = TemporalNorm(embed_dim)

        # Learnable symbols for abstractors
        self.symbols_abs = nn.Parameter(torch.randn(num_symbols_abs, embed_dim * symbol_factor_abs))
        self.symbols_ternary = nn.Parameter(torch.randn(num_symbols_ternary, embed_dim * symbol_factor_tern))

        # self.pos_embed_tern = nn.Parameter(torch.randn(num_symbols_ternary, embed_dim))

        # Fixed 2D sinusoidal embeddings for the ternary row/column embeddings
        # self.pos_embed_tern = nn.Parameter(torch.zeros([num_symbols_ternary, embed_dim]), requires_grad=False)
        # pos_embed_data_tern = pos.get_2d_sincos_pos_embed_rect(embed_dim=embed_dim, grid_height=2, grid_width=3, cls_token=False)
        # self.pos_embed_tern.data.copy_(torch.from_numpy(pos_embed_data_tern).float())

        # Abstractor layers
        self.abstractor = nn.ModuleList([  # Abstractor layers
            Block(embed_dim * (1 if i == 0 else symbol_factor_abs), embed_dim * symbol_factor_abs, abs_num_heads,
                  abs_mlp_ratio, abs_proj_drop, abs_attn_drop, abs_drop_path_max * ((i + 1) / abs_depth),
                  norm_layer=norm_layer)
            for i in range(abs_depth)
        ])

        # Ternary layers
        self.ternary_module = nn.ModuleList([  # Ternary layers
            Block(embed_dim * (1 if i == 0 else symbol_factor_tern), embed_dim * symbol_factor_tern, tern_num_heads,
                  tern_mlp_ratio, tern_proj_drop, tern_attn_drop,
                  tern_drop_path_max * ((i + 1) / ternary_depth), norm_layer=norm_layer)
            for i in range(ternary_depth)
        ])

        # Transformer layers
        self.transformer = nn.ModuleList([  # Transformer layers
            Block(embed_dim, embed_dim, trans_num_heads, trans_mlp_ratio, trans_proj_drop, trans_attn_drop, trans_drop_path_max * ((i + 1) / trans_depth), norm_layer=norm_layer)
            for i in range(trans_depth)
        ])

        # Guesser head
        self.guesser_head = nn.Sequential(
            nn.LayerNorm(embed_dim + embed_dim * symbol_factor_abs + embed_dim * symbol_factor_tern,
                         eps=1e-5, elementwise_affine=True),
            nn.Linear(embed_dim + embed_dim * symbol_factor_abs + embed_dim * symbol_factor_tern, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # Ternary operation MLP
        # over-powered
        # self.phi_mlp = nn.Sequential(
        #     nn.Linear(3 * embed_dim, 6 * embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(6 * embed_dim, 3 * embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(3 * embed_dim, embed_dim)
        # )

        # original
        self.phi_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, phi_mlp_hidden_dim * embed_dim),
            nn.ReLU(),
            nn.Linear(phi_mlp_hidden_dim * embed_dim, embed_dim)
        )

        # under-powered
        # self.phi_mlp = nn.Sequential(
        #     nn.Linear(3 * embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim)
        # )

    def ternary_mlp(self, x):
        """
        Perform the ternary operation Φ_MLP(x1, x2, x3) for rows and columns across the sequence.
        Input x is of shape (batch_size, 9, embed_dim)
        """
        batch_size = x.size(0)

        slice_idx = torch.tensor([0, 3, 6, 0, 1, 2])
        increment = torch.tensor([1, 1, 1, 3, 3, 3])

        # Extract x1, x2, x3 for all sliding windows
        x1 = x[:, slice_idx, :]  # Shape: (batch_size, 6, embed_dim)
        x2 = x[:, slice_idx + increment, :]  # Shape: (batch_size, 6, embed_dim)
        x3 = x[:, slice_idx + 2 * increment, :]  # Shape: (batch_size, 6, embed_dim)

        x1 = x1.reshape(batch_size * 6, -1)
        x2 = x2.reshape(batch_size * 6, -1)
        x3 = x3.reshape(batch_size * 6, -1)

        x = torch.cat([x1, x2, x3], dim=-1)

        # Apply the MLP
        result = self.phi_mlp(x)  # Shape: (batch_size * 6, embed_dim)

        return result.view(batch_size, 6, -1)

    def forward(self, sentences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sentences: Input tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        Returns:
            embeddings: Raw embeddings from the perception module.
            recreation: Reconstructed outputs after denormalization.
            scores: Candidate scores.
        """
        batch_size, num_candidates, grid_nodes, _, height, width = sentences.size()

        # Get embeddings and reconstructed sentences from Perception
        embeddings, reconstructed_sentences = self.perception.forward(sentences)

        # Add positional embeddings before normalization
        pos_embed = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, num_candidates, grid_nodes, -1)
        embeddings = embeddings + pos_embed  # Shape: [batch_size, num_candidates, grid_size**2, embed_dim]

        # Normalize embeddings
        embeddings_normalized = self.temporal_norm.forward(embeddings)

        # # Add positional embeddings after normalization
        # pos_embed = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, num_candidates, grid_nodes, -1)
        # embeddings_normalized = embeddings_normalized + pos_embed  # Shape: [batch_size, num_candidates, grid_size**2, embed_dim]

        # Reshape embeddings_normalized to ensure correct dimensionality before passing to ternary_mlp and beyond
        embeddings_normalized_reshaped = embeddings_normalized.view(batch_size * num_candidates, grid_nodes, self.embed_dim)

        # Apply ternary operation to obtain row/column embeddings
        ternary_tokens_unnormalized = self.ternary_mlp(embeddings_normalized_reshaped) # [batch_size * num_candidates, num_symbols_ternary, embed_dim]

        # Temporal context normalization of new row/column embeddings
        ternary_tokens_normalized = self.temporal_norm_tern.forward(ternary_tokens_unnormalized)
        ternary_tokens_unnorm_reshaped = ternary_tokens_unnormalized.view(
            [batch_size, self.num_candidates, self.grid_size * 2, -1]) # for use later in de-normalizing

        # add positional encodings to ternary tokens
        # pos_embed_tern = self.pos_embed_tern.unsqueeze(0).expand(batch_size * num_candidates,
        #                                                          self.num_symbols_ternary, -1)
        # ternary_tokens_normalized = ternary_tokens_normalized + pos_embed_tern

        # Temporarily expand the symbols for batch dimension manipulation
        expanded_symbols_abs = self.symbols_abs.unsqueeze(0).expand(batch_size * num_candidates, -1, -1)
        expanded_symbols_ternary = self.symbols_ternary.unsqueeze(0).expand(batch_size * num_candidates, -1, -1)

        # Process embeddings with abstractor
        abstracted = embeddings_normalized_reshaped.clone()
        for idx, blk in enumerate(self.abstractor):
            if idx == 0:
                abstracted = blk(
                    x_q=abstracted,
                    x_k=abstracted,
                    x_v=expanded_symbols_abs,
                )
            else:
                abstracted = blk(x_q=abstracted, x_k=abstracted, x_v=abstracted)

        # Process ternary tokens
        for idx, blk in enumerate(self.ternary_module):
            if idx == 0:
                ternary_tokens = blk(
                    x_q=ternary_tokens_normalized,
                    x_k=ternary_tokens_normalized,
                    x_v=expanded_symbols_ternary,
                )
            else:
                ternary_tokens = blk(x_q=ternary_tokens, x_k=ternary_tokens, x_v=ternary_tokens)

        # Process embeddings with transformer
        transformed = embeddings_normalized_reshaped.clone()
        for blk in self.transformer:
            transformed = blk(x_q=transformed, x_k=transformed, x_v=transformed)

        # Aggregating the three streams before scoring and recreation
        transformed = transformed.view([batch_size, self.num_candidates, self.grid_size ** 2, -1])
        abstracted = abstracted.view(batch_size, self.num_candidates, self.grid_size ** 2, -1)
        ternary_tokens = ternary_tokens.view(
            [batch_size, self.num_candidates, self.grid_size * 2, -1])

        # De-normalize the three streams (ternary_tokens_normalized, abstracted, transformed)
        transformed = self.temporal_norm.de_normalize(transformed, embeddings)

        if self.symbol_factor_abs == 1: # skip de-normalization when symbol_factor_abs != 1
            abstracted = self.temporal_norm.de_normalize(abstracted, embeddings)

        if self.symbol_factor_tern == 1: # skip de-normalization when symbol_factor_tern != 1
            ternary_tokens = self.temporal_norm_tern.de_normalize(ternary_tokens, ternary_tokens_unnorm_reshaped)

        trans_abs = torch.cat([transformed, abstracted], dim=-1)

        reas_bottleneck = torch.cat([trans_abs.mean(dim=-2), ternary_tokens.mean(dim=-2)], dim=-1).view(
            batch_size * self.num_candidates, -1)

        # Scores from the concatenated outputs
        scores = self.guesser_head(reas_bottleneck).view(batch_size, num_candidates)  # [batch_size, num_candidates]

        return reconstructed_sentences, scores


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
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        restrict_qk=False,
    ):
        super().__init__()

        assert dim_kq % num_heads == 0, "dim_kq should be divisible by num_heads"
        assert dim_v % num_heads == 0, "dim_v should be divisible by num_heads"

        self.dim_kq = dim_kq
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.head_dim_kq = dim_kq // num_heads
        self.head_dim_v = dim_v // num_heads
        self.scale = self.head_dim_kq**-0.5

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
        """
        Handles inputs with varying dimensions by flattening and reshaping.
        Args:
            x_q, x_k, x_v: Query, Key, and Value tensors.
        Returns:
            The output tensor after applying attention.
        """
        original_shape = x_v.shape
        batch_size = original_shape[0]
        len_q, len_k, len_v = x_q.shape[-2], x_k.shape[-2], x_v.shape[-2]

        # Multi-head attention reshaping
        q = self.w_qs(x_q).view(batch_size, len_q, self.num_heads, self.head_dim_kq).permute(0, 2, 1, 3)
        k = self.w_ks(x_k).view(batch_size, len_k, self.num_heads, self.head_dim_kq).permute(0, 2, 1, 3)
        v = self.w_vs(x_v).view(batch_size, len_v, self.num_heads, self.head_dim_v).permute(0, 2, 1, 3)

        # Normalize queries and keys (if enabled)
        if self.restrict_qk:
            q = self.qk_norm(q * self.scale)
            k = self.qk_norm(k)
        else:
            q = self.q_norm(q * self.scale)
            k = self.k_norm(k)

        # Compute scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1))  # (batch, num_heads, len_q, len_k)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)  # (batch, num_heads, len_q, head_dim_v)

        # Reshape and project back
        x = x.permute(0, 2, 1, 3).reshape(-1, len_q, self.dim_v)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore the original batch and candidate dimensions
        x = x.view(*original_shape[:-1], self.dim_v)

        return x


class TemporalNorm(nn.Module):
    """
    Temporal normalization layer normalizing across the temporal (grid/sequence) dimension.
    """
    def __init__(self, embed_dim, eps=1e-5):
        """
        Args:
            embed_dim (int): Dimensionality of the embeddings/features.
            eps (float): A small value to prevent division by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(embed_dim))  # Learnable shift
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, num_candidates, grid_size**2, embed_dim].
        Returns:
            Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(dim=2, keepdim=True)  # Mean across grid nodes
        std = x.std(dim=2, keepdim=True) + self.eps  # Standard deviation
        x_normalized = (x - mean) / std
        return x_normalized * self.weight + self.bias

    def de_normalize(self, x: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        De-normalizes the input tensor using the stored weight and bias.
        Args:
            x (Tensor): Normalized tensor of shape [batch_size, num_candidates, grid_size**2, embed_dim].
            original (Tensor): Original input tensor used for normalization.
        Returns:
            Tensor: De-normalized tensor with the same shape as input.
        """
        mean = original.mean(dim=2, keepdim=True)
        std = original.std(dim=2, keepdim=True) + self.eps
        return (x - self.bias) / self.weight * std + mean



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
        mlp_ratio=4.0,
        q_bias=False,
        k_bias=False,
        v_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        restrict_qk=False,
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
            restrict_qk=restrict_qk,
        )
        self.ls1 = LayerScale(dim_kq, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = mlp_layer(
            in_features=dim_v,
            hidden_features=int(dim_v * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim_v, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dim_kq = dim_kq
        self.dim_v = dim_v

    def forward(self, x_q, x_k, x_v, use_mlp_layer=True):
        """
        Handles self-attention inside the Block.
        Args:
            x_q, x_k, x_v: Query, Key, and Value tensors.
            use_mlp_layer (bool): Whether to apply the MLP layer.
        Returns:
            Tensor after self-attention and optional MLP layer.
        """
        batch_candidates, grid_nodes, _ = x_q.shape  # This is batch_size * num_candidates

        # Apply attention
        attn_output = self.attn.forward(self.norm1(x_q), self.norm1(x_k), self.norm1_v(x_v))

        # Ensure output shape matches x_v before summation
        assert attn_output.shape == x_v.shape, f"Shape mismatch: attn_output={attn_output.shape}, x_v={x_v.shape}"
        x = x_v + self.drop_path1(self.ls1(attn_output))

        # Optional MLP layer
        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))

        return x  # Shape: (batch_size * num_candidates, grid_nodes, dim_v)

