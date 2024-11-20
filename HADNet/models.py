import pos_embed as pos
import torch
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
                 per_mlp_drop=0):
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
        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=bb_depth, num_heads=bb_num_heads, mlp_drop=per_mlp_drop)

    def forward(self, sentences):
        """
        Args:
            sentences: Tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        Returns:
            embeddings_final: Tensor of shape [batch_size, num_candidates, grid_size**2, embed_dim]
        """
        batch_size = sentences.size(0)

        # Reshape for processing by BackbonePerception
        sentences_reshaped = sentences.view(-1, 1, 160, 160)  # Shape: [B * N_c * G^2, 1, 160, 160]
        features = self.perception.forward(sentences_reshaped)  # Shape: [B * N_c * G^2, embed_dim]

        # Reshape back to original context
        features_reshaped = features.view(batch_size, self.num_candidates, self.n_nodes, -1)  # [B, N_c, G^2, embed_dim]

        # Expand and add positional embeddings
        pos_embed_expanded = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_candidates, self.n_nodes, -1)
        embeddings_final = features_reshaped + pos_embed_expanded  # Element-wise addition

        return embeddings_final


class DialogicIntegrator(nn.Module):
    def __init__(self, embed_dim: int, n_levels: int):
        super().__init__()
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(n_levels)
        ])

        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, assertion: torch.Tensor, doubt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [batch_size * num_candidates, 9, embed_dim]
        batch_cand_size, n_nodes, embed_dim = assertion.shape

        uncertainties = []
        x_assertion = assertion
        x_doubt = doubt

        for level in self.levels:
            # Estimate uncertainty
            uncertainty = self.uncertainty_estimator(x_assertion)
            uncertainties.append(uncertainty)

            # Integrate
            combined = torch.cat([x_assertion, x_doubt], dim=-1)
            integrated = level(combined)
            x_assertion = uncertainty * integrated + (1 - uncertainty) * x_assertion

        uncertainties = torch.stack(uncertainties, dim=1).mean(dim=1)  # Shape: [batch_size * num_candidates, n_nodes, 1]
        return x_assertion, uncertainties


class HADNet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        grid_size: int = 3,
        num_candidates: int = 8,
        n_levels: int = 3,
        bb_depth: int = 2,
        bb_num_heads: int = 8,
        trans_depth: int = 2,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_max: float = 0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_candidates = num_candidates

        # Perception module
        self.perception = Perception(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_candidates=num_candidates,
            bb_depth=bb_depth,
            bb_num_heads=bb_num_heads,
        )

        # Temporal context normalization
        self.temporal_norm = TemporalNorm(embed_dim)

        # Holonic assertion-doubt streams
        self.assertion_stream = nn.ModuleList([
            Block(embed_dim, embed_dim, bb_num_heads, mlp_ratio=mlp_ratio,
                  proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path_max, norm_layer=norm_layer)
            for _ in range(trans_depth)
        ])

        self.doubt_stream = nn.ModuleList([
            Block(embed_dim, embed_dim, bb_num_heads, mlp_ratio=mlp_ratio,
                  proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path_max, norm_layer=norm_layer)
            for _ in range(trans_depth)
        ])

        # Integration
        self.integrator = DialogicIntegrator(embed_dim, n_levels)

        # Output heads
        self.recreation_head = nn.Linear(embed_dim, embed_dim)
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * grid_size * grid_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, sentences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sentences: Input tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        Returns:
            embeddings: Raw embeddings from the perception module.
            recreation: Recreated outputs after denormalization (flattened).
            scores: Candidate scores.
        """
        # Extract batch size and other dimensions
        batch_size = sentences.size(0)
        num_candidates = sentences.size(1)
        n_nodes = self.grid_size ** 2
        embed_dim = self.embed_dim

        # Pass through Perception module
        embeddings = self.perception.forward(sentences)  # Shape: [batch_size, num_candidates, grid_size**2, embed_dim]

        # Temporal normalization (single normalization shared for both streams)
        embeddings_normalized = self.temporal_norm.forward(embeddings)

        # Process through assertion and doubt transformer blocks
        x_assertion = embeddings_normalized
        x_doubt = embeddings_normalized

        for assert_block, doubt_block in zip(self.assertion_stream, self.doubt_stream):
            x_assertion = assert_block(x_q=x_assertion, x_k=x_assertion, x_v=x_assertion)
            x_doubt = doubt_block(x_q=x_doubt, x_k=x_doubt, x_v=x_doubt)

        # Reshape before integration
        x_assertion = x_assertion.view(batch_size * num_candidates, n_nodes, embed_dim)
        x_doubt = x_doubt.view(batch_size * num_candidates, n_nodes, embed_dim)

        # Integrate streams
        integrated, uncertainty = self.integrator.forward(x_assertion, x_doubt)

        # Reshape back after integration
        integrated = integrated.view(batch_size, num_candidates, n_nodes, embed_dim)

        # Generate recreation
        recreation = self.recreation_head(integrated)  # Shape: [batch_size, num_candidates, grid_size**2, embed_dim]

        # De-normalize the recreation
        recreation_de_normalized = self.temporal_norm.de_normalize(recreation, embeddings)

        # Reshape recreation to match flattened embeddings
        recreation_flattened = recreation_de_normalized.view(batch_size * num_candidates, n_nodes, embed_dim)

        # Flatten nodes for scoring
        flat_integrated = integrated.view(batch_size * num_candidates,
                                          -1)  # [batch_size * num_candidates, grid_size**2 * embed_dim]
        scores = self.score_head(flat_integrated).view(batch_size, num_candidates)  # [batch_size, num_candidates]

        return embeddings, recreation_flattened, scores


class ReasoningModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        grid_size: int,
        abs_depth: int,
        trans_depth: int,
        ternary_depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_max: float = 0.0,
        num_symbols_abs: int = 9,
        num_symbols_ternary: int = 6,
        norm_layer=nn.LayerNorm,
        bb_depth: int = 2,
        bb_num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Initialize Perception module
        self.perception = Perception(
            embed_dim=embed_dim,
            grid_size=grid_size,
            num_candidates=8,  # Default assumed; adapt as needed
            bb_depth=bb_depth,
            bb_num_heads=bb_num_heads,
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed_data = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float())

        # Temporal normalization
        self.temporal_norm = TemporalNorm(embed_dim)

        # Learnable symbols for abstractors
        self.symbols_abs = nn.Parameter(torch.randn(num_symbols_abs, embed_dim))
        self.symbols_ternary = nn.Parameter(torch.randn(num_symbols_ternary, embed_dim))

        # Abstractor layers
        self.abstractor = nn.ModuleList([  # Abstractor layers
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, proj_drop, attn_drop, drop_path_max * ((i + 1) / abs_depth), norm_layer=norm_layer)
            for i in range(abs_depth)
        ])

        # Ternary layers
        self.ternary_module = nn.ModuleList([  # Ternary layers
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, proj_drop, attn_drop, drop_path_max * ((i + 1) / ternary_depth), norm_layer=norm_layer)
            for i in range(ternary_depth)
        ])

        # Transformer layers
        self.transformer = nn.ModuleList([  # Transformer layers
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, proj_drop, attn_drop, drop_path_max * ((i + 1) / trans_depth), norm_layer=norm_layer)
            for i in range(trans_depth)
        ])

        # Reconstruction decoder (adjusted to handle concatenated inputs)
        self.decoder = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),  # Assuming concatenated will have 3 * embed_dim
            nn.ReLU(),
            nn.Linear(embed_dim, grid_size**2 * embed_dim),
            nn.Sigmoid(),
        )

        # Guesser head
        self.guesser_head = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # Ternary operation MLP
        self.phi_mlp = nn.Sequential(
            nn.Linear(3 * embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

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

    def forward(self, sentences: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sentences: Input tensor of shape [batch_size, num_candidates, grid_size**2, 1, 160, 160]
        Returns:
            embeddings: Raw embeddings from the perception module.
            recreation: Reconstructed outputs after denormalization.
            scores: Candidate scores.
        """
        batch_size, num_candidates, grid_nodes, _, height, width = sentences.size()

        # Reshape sentences for perception
        embeddings = self.perception.forward(sentences)  # Shape: [batch_size, num_candidates, grid_nodes, embed_dim]

        # Add positional embeddings
        pos_embed = self.pos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, num_candidates, grid_nodes, -1)
        embeddings = embeddings + pos_embed  # Shape: [batch_size, num_candidates, grid_size**2, embed_dim]

        # Normalize embeddings
        embeddings_normalized = self.temporal_norm.forward(embeddings)

        # Reshape embeddings_normalized to ensure correct dimensionality before passing to ternary_mlp
        embeddings_normalized_reshaped = embeddings_normalized.view(batch_size * num_candidates, grid_nodes, self.embed_dim)

        # Now apply ternary operation
        ternary_tokens = self.ternary_mlp(embeddings_normalized_reshaped)
        ternary_tokens_normalized = self.temporal_norm.forward(ternary_tokens)

        # Temporarily expand the symbols for batch dimension manipulation
        expanded_symbols_abs = self.symbols_abs.unsqueeze(0).expand(batch_size * num_candidates, -1, -1)
        expanded_symbols_ternary = self.symbols_ternary.unsqueeze(0).expand(batch_size * num_candidates, -1, -1)

        # Process embeddings with abstractor
        abstracted = embeddings_normalized
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
                ternary_tokens_normalized = blk(
                    x_q=ternary_tokens_normalized,
                    x_k=ternary_tokens_normalized,
                    x_v=expanded_symbols_ternary,
                )
            else:
                ternary_tokens_normalized = blk(x_q=ternary_tokens_normalized, x_k=ternary_tokens_normalized, x_v=ternary_tokens_normalized)

        # Process embeddings with transformer
        transformed = embeddings_normalized.clone()
        for blk in self.transformer:
            transformed = blk(x_q=transformed, x_k=transformed, x_v=transformed)

        # De-normalize the three streams (ternary_tokens_normalized, abstracted, transformed)
        ternary_tokens = self.temporal_norm.de_normalize(ternary_tokens_normalized, ternary_tokens)
        abstracted = self.temporal_norm.de_normalize(abstracted, embeddings)
        transformed = self.temporal_norm.de_normalize(transformed, embeddings)

        # Aggregating the three streams before scoring and recreation
        transformed = transformed.view([batch_size, self.num_candidates, self.grid_size ** 2, -1])
        abstracted = abstracted.view(batch_size, self.num_candidates, self.grid_size ** 2, -1)
        ternary_tokens = ternary_tokens.view([batch_size, self.num_candidates, self.grid_size * 2, -1])

        trans_abs = torch.cat([transformed, abstracted], dim=-1)

        reas_bottleneck = torch.cat([trans_abs.mean(dim=-2), ternary_tokens.mean(dim=-2)], dim=-1).view(
            batch_size * self.num_candidates, -1)

        # Reconstruct the input from the concatenated outputs
        recreation = self.decoder(reas_bottleneck)  # Shape: [batch_size * num_candidates, grid_size**2 * embed_dim]
        recreation = recreation.view(batch_size, self.grid_size**2, -1)  # Shape: [batch_size, grid_size**2, embed_dim]

        # Scores from the concatenated outputs
        scores = self.guesser_head(reas_bottleneck).view(batch_size, num_candidates)  # [batch_size, num_candidates]

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
        # Flatten additional dimensions before processing
        original_shape = x_q.shape
        batch_size = original_shape[0]
        len_q = original_shape[-2]
        c = original_shape[-1]

        # Flatten for multi-head attention
        x_q = x_q.view(-1, len_q, c)
        x_k = x_k.view(-1, len_q, c)
        x_v = x_v.view(-1, len_q, c)

        # Multi-head attention reshaping
        q = self.w_qs(x_q).view(-1, len_q, self.num_heads, self.head_dim_kq).permute(0, 2, 1, 3)
        k = self.w_ks(x_k).view(-1, len_q, self.num_heads, self.head_dim_kq).permute(0, 2, 1, 3)
        v = self.w_vs(x_v).view(-1, len_q, self.num_heads, self.head_dim_v).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v)

        # Reshape and project back
        x = x.permute(0, 2, 1, 3).reshape(-1, len_q, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore the original batch and candidate dimensions
        x = x.view(*original_shape[:-1], c)

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

    def forward(self, x_q, x_k, x_v, use_mlp_layer=True):
        # Check if the input is flattened (3D) or not (4D)
        reshaped = False
        if len(x_q.shape) == 3:
            # If 3D, reshape to 4D
            batch_size, grid_nodes, embed_dim = x_q.size()
            x_q = x_q.view(batch_size, 1, grid_nodes, embed_dim)
            x_k = x_k.view(batch_size, 1, grid_nodes, embed_dim)
            x_v = x_v.view(batch_size, 1, grid_nodes, embed_dim)
            reshaped = True

        # Apply attention
        x = x_v + self.drop_path1(
            self.ls1(self.attn.forward(self.norm1(x_q), self.norm1(x_k), self.norm1_v(x_v)))
        )

        # Optional MLP layer
        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))

        # If the input was originally 3D, reshape back to 3D
        if reshaped:
            batch_size, _, grid_nodes, embed_dim = x.size()
            x = x.view(batch_size, grid_nodes, embed_dim)

        return x
