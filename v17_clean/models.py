import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn

class TransformerModelv19(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=768,
                 symbol_factor = 1,
                 grid_size = 3,
                 trans_num_heads=64,
                 abs_1_num_heads=64,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 trans_depth = 4,
                 abs_1_depth = 4,
                 cat_pos=True,
                 use_backbone = True,
                 bb_depth = 4,
                 bb_num_heads = 32):

        super(TransformerModelv19, self).__init__()

        assert abs_1_depth >= 2, 'Abstractor 1 depth must be at least 2'

        self.cat_pos = cat_pos
        self.embed_dim = embed_dim
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.use_backbone = use_backbone
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads

        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.bb_depth, num_heads=bb_num_heads)\
            if self.use_backbone else ResNetEncoder(embed_dim=self.embed_dim)

        if self.cat_pos:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        self.tcn = TemporalContextNorm(num_features=self.model_dim)

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.relBottleneck_1 = Block(self.model_dim, self.model_dim * self.symbol_factor, abs_1_num_heads, mlp_ratio, \
                                   q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, \
                                   attn_drop=0.1)

        self.blocks_abs_1 = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor * 2, self.model_dim * self.symbol_factor * 2, abs_1_num_heads,\
                  mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, \
                  attn_drop=0.1, drop_path=0.5*((i+1)/abs_1_depth))
            for i in range(abs_1_depth-1)])

        self.blocks_trans = nn.ModuleList([
            Block(self.model_dim, self.model_dim, trans_num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5 * ((i + 1) / trans_depth))
            for i in range(trans_depth)])

        self.norm_x = norm_layer(self.model_dim * self.symbol_factor * 2)

        self.norm_y = norm_layer(self.model_dim)

        self.mlp1 = nn.Linear(self.model_dim + self.model_dim * self.symbol_factor * 2, self.embed_dim)

        self.relu = nn.ReLU()

        self.mlp2 = nn.Linear(self.embed_dim, 1)

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim)

        # define symbols
        normal_initializer = torch.nn.init.normal_
        self.symbols_1 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))
        self.symbols_2 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))

    def ternary_operation(self, x):
        """
        Perform the ternary operation C(x1, x2, x3) for sliding windows across the sequence.
        Input x is of shape (batch_size, 9, embed_dim), and we pad the sequence by repeating
        the last vector twice to handle edge cases.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Ensure seq_len is sufficient for at least one sliding window
        assert seq_len >= 3, "Sequence length must be at least 3 for a sliding window."

        # Pad the sequence by repeating the last vector twice
        padding = x[:, -1, :].unsqueeze(1).repeat(1, 2, 1)  # Shape: (batch_size, 2, embed_dim)
        padded_x = torch.cat([x, padding], dim=1)  # Shape: (batch_size, seq_len + 2, embed_dim)

        # Extract x1, x2, x3 for all sliding windows
        x1 = padded_x[:, :-2, :].unsqueeze(2)  # Shape: (batch_size, seq_len-2, 1, embed_dim)
        x2 = padded_x[:, 1:-1, :].unsqueeze(3)  # Shape: (batch_size, seq_len-2, embed_dim, 1)
        x3 = padded_x[:, 2:, :].unsqueeze(3)  # Shape: (batch_size, seq_len-2, embed_dim, 1)

        # Compute the outer product
        outer_product = torch.matmul(x1, x2)  # Shape: (batch_size, seq_len-2, embed_dim, embed_dim)

        # Matrix-vector multiplication on the last two dimensions
        result = torch.matmul(outer_product, x3)

        # Squeeze to remove singleton dimension
        result = result.squeeze(-1)  # Shape: (batch_size, seq_len-2, embed_dim)

        return result

    def forward(self, sentences):
        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # x is (B, 8, 9, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped) # x_reshaped is (B*9*8, embed_dim)
        x = embed_reshaped.view(batch_size, 8, 9, -1) # x is (B, 8, 9, embed_dim)

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, 8, -1, -1) # expand to fit batch (B, 8, 9, embed_dim)

        if self.cat_pos:
            x = torch.cat([x, final_pos_embed], dim=3)  # add positional embeddings
        else:
            x = x + final_pos_embed  # add positional embeddings

        x = self.tcn(x)

        # repeat symbols along batch dimension
        symbols_1 = self.symbols_1.unsqueeze(0)
        symbols_1 = symbols_1.repeat(batch_size*8, 1, 1)

        x_reshaped = x.view(-1, 9, self.model_dim)  # x is (B, 8, 9, self.model_dim)

        y_reshaped = x_reshaped.clone()

        x_reshaped = self.relBottleneck_1.forward(x_q=x_reshaped, x_k=x_reshaped, x_v=symbols_1)

        x_ternary = self.ternary_operation(x_reshaped)
        x_reshaped_1 = torch.cat([x_reshaped, x_ternary], dim=-1)

        for blk in self.blocks_abs_1: # multi-headed self-attention layer
            x_reshaped_1 = blk(x_q=x_reshaped_1, x_k=x_reshaped_1, x_v=x_reshaped_1)
        x_reshaped_1 = self.norm_x(x_reshaped_1)

        # reduce dimension from symbol dimensions to embedding dimensions if self.cat_output = False
        x = x_reshaped_1.view([batch_size, 8, 9, -1])

        for blk in self.blocks_trans: # multi-headed self-attention layer
            y_reshaped = blk(x_q=y_reshaped, x_k=y_reshaped, x_v=y_reshaped)
        y_reshaped = self.norm_y(y_reshaped)

        y = y_reshaped.view(batch_size, 8, 9, -1)

        y = self.tcn.inverse(y)

        z = torch.cat([x,y], dim=3)

        # z_reshaped = z[:,:,8,:].view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        z_reshaped = torch.mean(z, dim=2).view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        dist_reshaped = self.mlp2(self.relu(self.mlp1(z_reshaped))) # dist_reshaped is (B*8, 1)

        dist = dist_reshaped.view(batch_size, 8)

        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation

    def encode(self, images):
        embeddings = self.perception.forward(images) # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings) # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images

class TransformerModelv18(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=768,
                 symbol_factor = 1,
                 grid_size = 3,
                 trans_num_heads=64,
                 abs_1_num_heads=64,
                 abs_2_num_heads=64,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 trans_depth = 4,
                 abs_1_depth = 4,
                 abs_2_depth = 4,
                 cat_pos=True,
                 cat_output=True,
                 use_backbone = True,
                 bb_depth = 4,
                 bb_num_heads = 32):

        super(TransformerModelv18, self).__init__()

        assert abs_1_depth >= 2, 'Abstractor 1 depth must be at least 2'
        assert abs_2_depth >= 2, 'Abstractor 2 depth must be at least 2'

        self.cat_pos = cat_pos
        self.cat_output = cat_output
        self.embed_dim = embed_dim
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.use_backbone = use_backbone
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads

        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.bb_depth, num_heads=bb_num_heads)\
            if self.use_backbone else ResNetEncoder(embed_dim=self.embed_dim)

        if self.cat_pos:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        self.tcn = TemporalContextNorm(num_features=self.model_dim)

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.relBottleneck_1 = Block(self.model_dim, self.model_dim * self.symbol_factor, abs_1_num_heads, mlp_ratio, \
                                   q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, \
                                   attn_drop=0.1)

        self.blocks_abs_1 = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_1_num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5*((i+1)/abs_1_depth))
            for i in range(abs_1_depth-1)])

        self.relBottleneck_2 = Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, \
                                     abs_2_num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, \
                                     norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1)

        self.blocks_abs_2 = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_2_num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5 * ((i + 1) / abs_2_depth))
            for i in range(abs_2_depth - 1)])

        self.blocks_trans = nn.ModuleList([
            Block(self.model_dim, self.model_dim, trans_num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5 * ((i + 1) / trans_depth))
            for i in range(trans_depth)])

        self.norm_x = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_y = norm_layer(self.model_dim)

        self.mlp1 = nn.Linear(self.model_dim * self.symbol_factor, self.model_dim) if not self.cat_output \
            else nn.Identity()

        if self.cat_output:
            self.mlp2 = nn.Linear(self.model_dim + self.model_dim * self.symbol_factor, self.embed_dim)
        else:
            self.mlp2 = nn.Linear(self.model_dim, self.embed_dim)

        self.relu = nn.ReLU()

        self.mlp3 = nn.Linear(self.embed_dim, 1)

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim)

        # define symbols
        normal_initializer = torch.nn.init.normal_
        self.symbols_1 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))
        self.symbols_2 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))

    def forward(self, sentences):
        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # x is (B, 8, 9, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped) # x_reshaped is (B*9*8, embed_dim)
        x = embed_reshaped.view(batch_size, 8, 9, -1) # x is (B, 8, 9, embed_dim)

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, 8, -1, -1) # expand to fit batch (B, 8, 9, embed_dim)

        if self.cat_pos:
            x = torch.cat([x, final_pos_embed], dim=3)  # add positional embeddings
        else:
            x = x + final_pos_embed  # add positional embeddings

        x = self.tcn(x)

        # repeat symbols along batch dimension
        symbols_1 = self.symbols_1.unsqueeze(0)
        symbols_1 = symbols_1.repeat(batch_size*8, 1, 1)

        symbols_2 = self.symbols_2.unsqueeze(0)
        symbols_2 = symbols_2.repeat(batch_size * 8, 1, 1)

        x_reshaped = x.view(-1, 9, self.model_dim)  # x is (B, 8, 9, self.model_dim)

        y_reshaped = x_reshaped.clone()

        x_reshaped = self.relBottleneck_1.forward(x_q=x_reshaped, x_k=x_reshaped, x_v=symbols_1)

        for blk in self.blocks_abs_1: # multi-headed self-attention layer
            x_reshaped = blk(x_q=x_reshaped, x_k=x_reshaped, x_v=x_reshaped)
        x_reshaped = self.norm_x(x_reshaped)

        x_reshaped_1 = self.relBottleneck_2.forward(x_q=x_reshaped, x_k=x_reshaped, x_v=symbols_2)

        for blk in self.blocks_abs_2: # multi-headed self-attention layer
            x_reshaped_1 = blk(x_q=x_reshaped_1, x_k=x_reshaped_1, x_v=x_reshaped_1)
        x_reshaped_1 = self.norm_x(x_reshaped_1)

        # reduce dimension from symbol dimensions to embedding dimensions if self.cat_output = False
        # try additional higher-order reasoning blocks with skip connection
        x_reshaped_2 = self.mlp1(x_reshaped + x_reshaped_1)

        x = x_reshaped_2.view([batch_size, 8, 9, -1])

        for blk in self.blocks_trans: # multi-headed self-attention layer
            y_reshaped = blk(x_q=y_reshaped, x_k=y_reshaped, x_v=y_reshaped)
        y_reshaped = self.norm_y(y_reshaped)

        y = y_reshaped.view(batch_size, 8, 9, -1)

        y = self.tcn.inverse(y)

        if self.cat_output:
            z = torch.cat([x,y], dim=3)
        else:
            z = x + y

        # z_reshaped = z[:,:,8,:].view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        z_reshaped = torch.mean(z, dim=2).view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        dist_reshaped = self.mlp3(self.relu(self.mlp2(z_reshaped))) # dist_reshaped is (B*8, 1)

        dist = dist_reshaped.view(batch_size, 8)

        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation

    def encode(self, images):
        embeddings = self.perception.forward(images) # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings) # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images

class TransformerModelv17(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=768,
                 symbol_factor = 1,
                 grid_size = 3,
                 num_heads=32,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 depth = 5,
                 cat_pos=True,
                 cat_output=True,
                 use_backbone = True,
                 backbone_depth = 4):

        super(TransformerModelv17, self).__init__()

        assert depth >= 2, 'depth must be at least 2'

        self.cat_pos = cat_pos
        self.cat_output = cat_output
        self.embed_dim = embed_dim
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.use_backbone = use_backbone
        self.backbone_depth = backbone_depth

        self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.backbone_depth) if self.use_backbone else \
            ResNetEncoder(embed_dim=self.embed_dim)

        if self.cat_pos:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        self.tcn = TemporalContextNorm(num_features=self.model_dim)

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.relBottleneck = Block(self.model_dim, self.model_dim * self.symbol_factor, num_heads, mlp_ratio, \
                                   q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, \
                                   attn_drop=0.1)

        self.blocks_symbol = nn.ModuleList([
            Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5*((i+1)/depth))
            for i in range(depth-1)])

        self.blocks_embed = nn.ModuleList([
            Block(self.model_dim, self.model_dim, num_heads, mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5 * ((i + 1) / depth))
            for i in range(depth)])

        self.norm_x = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_y = norm_layer(self.model_dim)

        self.mlp1 = nn.Linear(self.model_dim * self.symbol_factor, self.model_dim) if not self.cat_output \
            else nn.Identity()

        if self.cat_output:
            self.mlp2 = nn.Linear(self.model_dim + self.model_dim * self.symbol_factor, self.embed_dim)
        else:
            self.mlp2 = nn.Linear(self.model_dim, self.embed_dim)

        self.relu = nn.ReLU()

        self.mlp3 = nn.Linear(self.embed_dim, 1)

        self.decoder = ResNetDecoder(embed_dim=self.embed_dim)

        normal_initializer = torch.nn.init.normal_
        self.symbols = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))

    def forward(self, sentences):
        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # x is (B, 8, 9, 1, 160, 160)
        embed_reshaped = self.perception.forward(sen_reshaped) # x_reshaped is (B*9*8, embed_dim)
        x = embed_reshaped.view(batch_size, 8, 9, -1) # x is (B, 8, 9, embed_dim)

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, 8, -1, -1) # expand to fit batch (B, 8, 9, embed_dim)

        if self.cat_pos:
            x = torch.cat([x, final_pos_embed], dim=3)  # add positional embeddings
        else:
            x = x + final_pos_embed  # add positional embeddings

        x = self.tcn(x)

        # repeat symbols along batch dimension
        symbols = self.symbols.unsqueeze(0)
        symbols = symbols.repeat(batch_size*8, 1, 1)

        x_reshaped = x.view(-1, 9, self.model_dim)  # x is (B, 8, 9, self.model_dim)

        y_reshaped = x_reshaped.clone()

        x_reshaped = self.relBottleneck(x_q=x_reshaped, x_k=x_reshaped, x_v=symbols)

        for blk in self.blocks_symbol: # multi-headed self-attention layer
            x_reshaped = blk(x_q=x_reshaped, x_k=x_reshaped, x_v=x_reshaped)
        x_reshaped = self.norm_x(x_reshaped)

        # reduce dimension from symbol dimensions to embedding dimensions if self.cat_output = False
        x_reshaped = self.mlp1(x_reshaped)

        x = x_reshaped.view([batch_size, 8, 9, -1])

        for blk in self.blocks_embed: # multi-headed self-attention layer
            y_reshaped = blk(x_q=y_reshaped, x_k=y_reshaped, x_v=y_reshaped)
        y_reshaped = self.norm_y(y_reshaped)

        y = y_reshaped.view(batch_size, 8, 9, -1)

        y = self.tcn.inverse(y)

        if self.cat_output:
            z = torch.cat([x,y], dim=3)
        else:
            z = x + y

        # z_reshaped = z[:,:,8,:].view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        z_reshaped = torch.mean(z, dim=2).view(batch_size * 8, -1) # z is (B, 8, 9, -1)
        dist_reshaped = self.mlp3(self.relu(self.mlp2(z_reshaped))) # dist_reshaped is (B*8, 1)

        dist = dist_reshaped.view(batch_size, 8)

        recreation = self.decoder.forward(embed_reshaped).view(batch_size, 8, 9, 1, 160, 160)

        return dist, recreation

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
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class BackbonePerception(nn.Module):
    def __init__(self, embed_dim, num_heads = 32, mlp_ratio = 4, norm_layer=nn.LayerNorm, depth = 4):
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
        )

        self.blocks = nn.ModuleList([
            Block(256, 256, self.num_heads, self.mlp_ratio, \
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, \
                  drop_path=0.5*((i+1)/self.depth)) for i in range(self.depth)])

        self.mlp = nn.Linear(256*10*10, self.embed_dim)

    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, 256, 10*10)
        x = x.transpose(1,2)

        for block in self.blocks:
            x = block(x_q = x, x_k = x, x_v = x)

        x = x.reshape(batch_dim, 256*10*10)

        x = self.mlp(x)

        return x

class ResNetDecoder(nn.Module):
    def __init__(self, embed_dim=768):
        super(ResNetDecoder, self).__init__()

        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 256*10*10),
            nn.Unflatten(1, (256,10,10)),
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
        return self.decoder(x)

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
            mlp_layer=Mlp,
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

        x = x_v + self.drop_path1(self.ls1(self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1_v(x_v))))

        if use_mlp_layer:
            x = self.norm3(x + self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        else:
            x = self.norm2(x)

        return x