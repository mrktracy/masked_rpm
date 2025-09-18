import numpy as np


def get_1d_sincos_pos_embed(embed_dim, sequence_length, cls_token=True):
    """
    Generate 1D sinusoidal positional encodings for a sequence of tokens.

    Args:
        embed_dim (int): Dimension of the embedding vector.
        sequence_length (int): Number of tokens in the sequence.

    Returns:
        np.ndarray: Positional encodings of shape (sequence_length, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even for sinusoidal encoding."

    positions = np.arange(sequence_length)[:, np.newaxis]  # Shape: (sequence_length, 1)
    omega = np.arange(embed_dim // 2)[np.newaxis, :] / (embed_dim / 2.)
    omega = 1. / (10000**omega)  # Shape: (1, embed_dim // 2)

    pos_enc = np.concatenate([
        np.sin(positions * omega),  # (sequence_length, embed_dim // 2)
        np.cos(positions * omega)   # (sequence_length, embed_dim // 2)
    ], axis=1)  # Shape: (sequence_length, embed_dim)

    if cls_token:
        pos_enc = np.concatenate([np.zeros([1, embed_dim]), pos_enc], axis=0)

    return pos_enc


def get_2d_sincos_pos_embed_rect(embed_dim, grid_height, grid_width, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# 2 Positional encodings: from FAIR's 'Masked Autoencoders...'
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


