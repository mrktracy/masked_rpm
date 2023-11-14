import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn

class TransformerModelv10(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self, embed_dim=768, grid_size = 3, num_heads=32, \
                 mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 4, cat=False):
        super(TransformerModelv10, self).__init__()

        self.cat = cat
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.perception = ResNetEncoder(embed_dim=embed_dim)

        if self.cat:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            # Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, \
            #       norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1, drop_path=0.5*((i+1)/depth))
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, \
                  norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(self.model_dim)

        self.decoder = ResNetDecoder(embed_dim=embed_dim)

    def forward(self, ims, mask_tensor):
        batch_size = ims.size(0)  # Get the batch size from the first dimension of x

        ims_reshaped = ims.view(-1, 1, 160, 160)  # x is (B, 9, 1, 160, 160)
        x_reshaped = self.perception(ims_reshaped)
        x = x_reshaped.view(batch_size, 9, -1) # x is (B, 9, embed_dim)

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1) # expand to fit batch (B, 9, embed_dim)

        if self.cat:
            x = torch.cat([x, final_pos_embed], dim=2)  # add positional embeddings
            mask_tensor = torch.cat([mask_tensor, mask_tensor], dim=2)
        else:
            x = x + final_pos_embed  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x_q=x, x_k=x, x_v=x)
        x = self.norm(x)

        guess = torch.sum(x*mask_tensor, dim=1) # make guess shape (B, model_dim)
        if self.cat:
            guess = guess[:,:self.embed_dim] # take only the first embed_dim as guess

        guess = self.decoder(guess) # create an image

        return guess

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

class TransformerModelv9(nn.Module):
    def __init__(self, embed_dim=768, grid_size = 3, num_heads=32, \
                 mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 20, cat=False):
        super(TransformerModelv9, self).__init__()

        self.cat = cat
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        if self.cat:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, \
                  norm_layer=norm_layer, drop_path=0.5*((i+1)/depth))
            for i in range(depth)])

        self.norm = norm_layer(self.model_dim)

        self.sig = nn.Sigmoid()

    def forward(self, x, mask_tensor):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1) # expand to fit batch (B, 9, embed_dim)

        if self.cat:
            x = torch.cat([x, final_pos_embed], dim=2)  # add positional embeddings
            mask_tensor = torch.cat([mask_tensor, mask_tensor], dim=2)
        else:
            x = x + final_pos_embed  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x_q=x, x_k=x, x_v=x)
        x = self.norm(x)

        guess = torch.sum(x*mask_tensor, dim=1) # make guess shape (batch_size, model_dim)
        if self.cat:
            guess = guess[:,:self.embed_dim] # take only the first embed_dim as guess

        guess = self.sig(guess) # put on same scale as targets

        return guess

class TransformerModelv8(nn.Module):
    def __init__(self, embed_dim=768, grid_size = 3, num_heads=32, \
                 mlp_ratio=4.,norm_layer=nn.LayerNorm, depth = 20, cat=False):
        super(TransformerModelv8, self).__init__()

        self.cat = cat
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        if self.cat:
            self.model_dim = 2*self.embed_dim
        else:
            self.model_dim = self.embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, \
                  norm_layer=norm_layer, drop_path=0.5*((i+1)/depth))
            for i in range(depth)])

        self.norm = norm_layer(self.model_dim)

        self.sig = nn.Sigmoid()

    def forward(self, x, first_patch=None):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        final_pos_embed = self.pos_embed.unsqueeze(0).expand(batch_size, -1, -1) # expand to fit batch (B, 9, embed_dim)
        if first_patch is not None:
            pad = torch.zeros(self.embed_dim)  # create padding token (1, embed_dim)
            int_pos_embed = final_pos_embed.clone() # (B, 9, embed_dim)
            for i in range(batch_size):
                if first_patch[i] > 0:
                    final_pos_embed[i,:first_patch[i],:] = pad # pad up to the first patch
                    final_pos_embed[i,first_patch[i]:,:] = \
                        int_pos_embed[i,:self.grid_size**2 - first_patch[i],:] # add positional embeddings from the first patch to the end

        if self.cat:
            x = torch.cat([x, final_pos_embed], dim=2)  # add positional embeddings
        else:
            x = x + final_pos_embed  # add positional embeddings

        for blk in self.blocks: # multi-headed self-attention layer
            x = blk(x_q=x, x_k=x, x_v=x)
        x = self.norm(x)

        guess = x[:,-1,:].squeeze() # make guess shape (batch_size, model_dim)
        guess = self.sig(guess) # put on same scale as targets

        return guess

class TransformerModelv7(nn.Module):
    def __init__(self, embed_dim=768, grid_size=3, num_heads=16, mlp_ratio=4., \
                 norm_layer=nn.LayerNorm, con_depth=10, can_depth=10, \
                 guess_depth=10, cat=False):
        super(TransformerModelv7, self).__init__()

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

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.lin = nn.Linear(self.model_dim, 1)

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

        y = candidates.clone()

        for blk1, blk2 in self.guess_blocks:
            y = blk1(x_q=y, x_k=y, x_v=y, use_mlp_layer = False)
            y = blk2(x_q=y, x_k=context, x_v=context)

        y_reshaped = y.view(-1, self.model_dim)
        guess_reshaped = self.lin(y_reshaped)
        guess = guess_reshaped.view(batch_size, 8)

        return guess

class TransformerModelv6(nn.Module):
    def __init__(self, embed_dim=256, grid_size=3, num_heads=32, mlp_ratio=4., norm_layer=nn.LayerNorm, con_depth=8,\
                 can_depth=8, guess_depth=8, cat=True):
        super(TransformerModelv6, self).__init__()

        self.cat = cat

        if self.cat == True:
            self.model_dim = 2*embed_dim
        else:
            self.model_dim = embed_dim

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.perception = ResNetEncoder(embed_dim=embed_dim)

        self.con_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(con_depth)])

        self.can_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(can_depth)])

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.lin = nn.Linear(self.model_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)  # Get the batch size from the first dimension of x

        x_reshaped = x.view(-1, 1, 160, 160)  # x is (B, 16, 1, 160, 160)
        y_reshaped = self.perception(x_reshaped)
        y = y_reshaped.view(batch_size, 16, -1)

        context = y[:,0:8,:] # x is (B, 16, embed_dim)
        candidates = y[:,8:,:]

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

        z = candidates.clone()

        for blk1, blk2 in self.guess_blocks:
            z = blk1(x_q=z, x_k=z, x_v=z, use_mlp_layer=False)
            z = blk2(x_q=z, x_k=context, x_v=context)

        z_reshaped = z.view(-1, self.model_dim)
        guess_reshaped = self.lin(z_reshaped)
        guess = guess_reshaped.view(batch_size, 8)

        return guess

class TransformerModelMNISTv6(nn.Module): # based on TransformerModelv6 but without positional embeddings
    def __init__(self, embed_dim=256, num_heads=16, mlp_ratio=4., \
                 norm_layer=nn.LayerNorm, con_depth=8, can_depth=8, guess_depth=8):
        super(TransformerModelMNISTv6, self).__init__()

        self.model_dim = embed_dim

        self.perception = ResNetEncoder(embed_dim=self.model_dim)

        self.con_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(con_depth)])

        self.can_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(can_depth)])

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth)
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

        z = candidates.clone()

        for blk1, blk2 in self.guess_blocks:
            z = blk1(x_q=z, x_k=z, x_v=z, use_mlp_layer=False)
            z = blk2(x_q=z, x_k=context, x_v=context)

        z_reshaped = z.view(-1,self.model_dim)
        guess_reshaped = self.lin(z_reshaped)
        guess = guess_reshaped.view(batch_size,8)

        return guess

class TransformerModelMNISTv3(nn.Module):
    def __init__(self, embed_dim=512, num_heads=16, mlp_ratio=4., \
                 norm_layer=nn.LayerNorm, con_depth=8, can_depth=8, guess_depth=8):
        super(TransformerModelMNISTv3, self).__init__()

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

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([grid_size**2, embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=embed_dim, grid_size=grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.perception = ResNetEncoder(embed_dim=embed_dim)

        self.abstr_blocks = nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)
            for _ in range(abstr_depth)])

        self.reas_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(reas_depth)
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

        if self.cat == True:
            context = torch.cat([context, self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)],\
                                dim=2)  # add positional embeddings
            candidates = torch.cat([candidates, self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)],\
                                dim=2)  # add 9th positional embedding to all candidates
        else:
            context = context + self.pos_embed[0:8].unsqueeze(0).expand(batch_size, -1, -1)  # add positional embeddings
            candidates = candidates + self.pos_embed[8].unsqueeze(0).expand(batch_size, 8, -1)  # add 9th positional embedding to all candidates

        y = torch.cat([context,candidates], dim=1)

        for blk in self.abstr_blocks: # multi-headed self-attention layer
            y = blk(x_q=y, x_k=y, x_v=y)

        context_enc = y[:, 0:8, :]  # x is (B, 16, embed_dim)
        candidates_enc = y[:, 8:, :]
        z = candidates_enc.clone()

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

        self.guess_blocks = nn.ModuleList([nn.ModuleList([
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer),
            Block(self.model_dim, num_heads, mlp_ratio, q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer)])
            for _ in range(guess_depth)
                             ])

        self.norm = norm_layer(self.model_dim)

        self.lin = nn.Linear(self.model_dim, 1)

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

        y = candidates.clone()

        for blk1, blk2 in self.guess_blocks:
            y = blk1(x_q=context, x_k=y, x_v=y)
            y = blk2(x_q=candidates, x_k=y, x_v=y)

        y_reshaped = y.view(-1, self.model_dim)
        guess_reshaped = self.lin(y_reshaped)
        guess = guess_reshaped.view(batch_size, 8)

        return guess

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