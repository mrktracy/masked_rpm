import pos_embed as pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
# from torch.jit import Final
from timm.layers import Mlp, DropPath, use_fused_attn
import logging

class TransformerModelv24(nn.Module): # takes in images, embeds, performs self-attention, and decodes to image
    def __init__(self,
                 embed_dim=512,
                 symbol_factor=1,
                 grid_size=3,
                 trans_num_heads=4,
                 abs_1_num_heads=4,
                 abs_2_num_heads=4,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 trans_depth=2,
                 abs_1_depth=2,
                 abs_2_depth=2,
                 use_backbone_enc=True,
                 decoder_num=1,
                 bb_depth=1,
                 bb_num_heads=2,
                 ternary_num=1,
                 mlp_drop=0.3,
                 proj_drop=0.3,
                 attn_drop=0.3,
                 drop_path_max=0.3,
                 per_mlp_drop=0.3,
                 ternary_drop=0.3,
                 ternary_mlp_ratio=1,
                 restrict_qk=False,
                 feedback_dim=1024,
                 meta_1_depth = 1,
                 meta_1_num_heads=2,
                 meta_1_attn_drop=0,
                 meta_1_proj_drop=0,
                 meta_1_drop_path_max=0,
                 meta_2_depth=1,
                 meta_2_num_heads=2,
                 meta_2_attn_drop=0,
                 meta_2_proj_drop=0,
                 meta_2_drop_path_max=0,
                 num_candidates=8,
                 score_rep=8,
                 num_loss_terms=3,
                 device=None
                 ):

        super(TransformerModelv24, self).__init__()

        self.embed_dim = embed_dim
        self.symbol_factor = symbol_factor
        self.grid_size = grid_size
        self.use_backbone_enc = use_backbone_enc
        self.decoder_num = decoder_num
        self.bb_depth = bb_depth
        self.bb_num_heads = bb_num_heads
        self.ternary_num = ternary_num
        self.restrict_qk = restrict_qk
        self.feedback_dim = feedback_dim
        self.feedback = None
        self.feedback_old = None
        self.num_candidates = num_candidates
        self.score_rep = score_rep
        self.device=device
        self.num_loss_terms = num_loss_terms

        if self.use_backbone_enc:
            if restrict_qk:
                self.perception = BackbonePerceptionAlt(embed_dim=self.embed_dim, depth=self.bb_depth,
                                                        num_heads=bb_num_heads,
                                                        mlp_drop=per_mlp_drop)
            else:
                self.perception = BackbonePerception(embed_dim=self.embed_dim, depth=self.bb_depth,
                                                     num_heads=bb_num_heads, mlp_drop=per_mlp_drop)
        else:
            self.perception = ResNetEncoder(embed_dim=self.embed_dim, mlp_drop=per_mlp_drop)

        self.model_dim = 2*self.embed_dim
        # self.model_dim = self.embed_dim

        self.tcn_1 = TemporalContextNorm(num_features=self.model_dim)
        self.tcn_2 = TemporalContextNorm(num_features=self.model_dim)

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size**2, self.embed_dim]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks_abs_1 = nn.ModuleList(
            [Block(self.model_dim, self.model_dim * self.symbol_factor, abs_1_num_heads,
                   mlp_ratio, q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                   attn_drop=attn_drop, drop_path=drop_path_max / abs_1_depth)] +
            [Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_1_num_heads,
                   mlp_ratio, q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                   attn_drop=attn_drop, drop_path=drop_path_max*((i+1)/abs_1_depth))
             for i in range(1, abs_1_depth)])

        self.blocks_abs_2 = nn.ModuleList(
            [Block(self.model_dim, self.model_dim * self.symbol_factor, abs_2_num_heads,
                   mlp_ratio, q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                   attn_drop=attn_drop, drop_path=drop_path_max / abs_2_depth)] +
            [Block(self.model_dim * self.symbol_factor, self.model_dim * self.symbol_factor, abs_2_num_heads,
                   mlp_ratio, q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                   attn_drop=attn_drop, drop_path=drop_path_max * ((i + 1) / abs_2_depth))
             for i in range(1, abs_2_depth)])

        if self.restrict_qk:
            self.blocks_trans = nn.ModuleList([
                Block(self.embed_dim, self.model_dim, trans_num_heads, mlp_ratio,
                      q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                      attn_drop=attn_drop, drop_path=drop_path_max * ((i + 1) / trans_depth), restrict_qk=self.restrict_qk)
                for i in range(trans_depth)])
        else:
            self.blocks_trans = nn.ModuleList([
                Block(self.model_dim, self.model_dim, trans_num_heads, mlp_ratio,
                      q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=proj_drop,
                      attn_drop=attn_drop, drop_path=drop_path_max * ((i + 1) / trans_depth), restrict_qk=self.restrict_qk)
                for i in range(trans_depth)])

        self.norm_x_1 = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_x_2 = norm_layer(self.model_dim * self.symbol_factor)

        self.norm_y = norm_layer(self.model_dim)

        # if incorporating meta-reasoning vector into guesser head, use this
        self.guesser_head = nn.Sequential(
            nn.Linear(self.model_dim + 2 * self.model_dim * self.symbol_factor + self.feedback_dim, self.embed_dim),
            nn.Dropout(p=mlp_drop),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1))

        # if not incorporating meta-reasoning vector into guesser head, use this
        # self.guesser_head = nn.Sequential(
        #     nn.Linear(self.model_dim + 2 * self.model_dim * self.symbol_factor, self.embed_dim),
        #     nn.Dropout(p=mlp_drop),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, 1))

        # # if not incorporating meta-reasoning vector into guesser head and using linear layer, use this
        # self.guesser_head = nn.Sequential(
        #     nn.Linear(self.model_dim + 2 * self.model_dim * self.symbol_factor, 1)
        # )

        if self.decoder_num == 1:
            self.decoder = MLPDecoder(embed_dim=self.embed_dim, mlp_drop=per_mlp_drop)
        elif self.decoder_num == 2:
            self.decoder = ResNetDecoder(embed_dim=self.embed_dim, mlp_drop=per_mlp_drop)
        else:
            self.decoder = BackboneDecoder(embed_dim=self.embed_dim, depth=self.bb_depth, num_heads=bb_num_heads,
                                           mlp_drop=per_mlp_drop)

        # define symbols
        normal_initializer = torch.nn.init.normal_
        self.symbols_1 = nn.Parameter(normal_initializer(torch.empty(9, self.model_dim * self.symbol_factor)))
        self.symbols_2 = nn.Parameter(normal_initializer(torch.empty(6, self.model_dim * self.symbol_factor)))

        # Define Φ_MLP for relation extraction
        self.phi_mlp = nn.Sequential(
            nn.Linear(3 * self.model_dim, ternary_mlp_ratio * self.model_dim),
            nn.ReLU(),
            nn.Linear(ternary_mlp_ratio * self.model_dim, self.model_dim),
            nn.Dropout(p=ternary_drop)
        )

        # if combining prior to positional encodings, use this
        self.combiner = nn.Sequential(
            nn.Linear(self.embed_dim + feedback_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # if combining after positional encodings, use this
        # self.combiner = nn.Sequential(
        #     nn.Linear(self.model_dim + feedback_dim, self.model_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.model_dim, self.model_dim)
        # )

        # # if combining after positional encodings and then re-affixing positional encodings, use this
        # self.combiner = nn.Sequential(
        #     nn.Linear(self.model_dim + feedback_dim, self.embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.embed_dim, self.embed_dim)
        # )

        self.reas_autoencoder = AutoencoderBottleneckAlt(input_dim=self.model_dim*3 + self.score_rep,
                                                         bottleneck_dim=self.feedback_dim,
                                                         depth=meta_1_depth,
                                                         num_heads=meta_1_num_heads,
                                                         mlp_ratio=mlp_ratio,
                                                         norm_layer=nn.LayerNorm,
                                                         proj_drop=meta_1_proj_drop,
                                                         attn_drop=meta_1_attn_drop,
                                                         drop_path_max=meta_1_drop_path_max,
                                                         seq_length=num_candidates)

        self.blocks_meta = nn.ModuleList([
            Block(self.feedback_dim, self.feedback_dim, meta_2_num_heads, mlp_ratio,
                  q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=meta_2_proj_drop,
                  attn_drop=meta_2_attn_drop, drop_path=meta_2_drop_path_max * ((i + 1) / meta_2_depth),
                  restrict_qk=False)
            for i in range(meta_2_depth)])

        # self.cls_token = nn.Parameter(torch.ones(self.feedback_dim)/self.feedback_dim)
        self.register_buffer('cls_token', torch.ones(self.feedback_dim))

        # self.perception_norm = L2Norm(dim=-1)
        # self.feedback_norm = L2Norm(dim=-1)

        self.loss_weight_mlp = LossWeightingMLP(feedback_dim=feedback_dim, num_loss_terms=num_loss_terms)

    def reset_feedback(self):
        self.feedback = None

    @staticmethod
    def ternary_operation(x):
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

    @staticmethod
    def ternary_hadamard(x):
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
        rules = self.phi_mlp(x)  # Shape: (batch_size * 6, embed_dim)

        # Matrix-vector multiplication on the last two dimensions
        result = rules.reshape(batch_size, 6, -1)

        return result

    def forward(self, sentences):

        batch_size = sentences.size(0)  # Get the batch size from the first dimension of x

        sen_reshaped = sentences.view(-1, 1, 160, 160)  # x is (B, 8, 9, 1, 160, 160)

        # logging.info("Begin perception...\n")

        embed_reshaped = self.perception.forward(sen_reshaped)  # x_reshaped is (B*9*8, embed_dim)
        embed_cached = embed_reshaped.clone()
        # embed_reshaped = self.perception_norm.forward(embed_reshaped)

        # attempt at new approach
        # add classification token for further processing across batch dimension
        if self.feedback is not None:
            cls_tokens = self.cls_token.unsqueeze(0)
            reas_encoded = torch.cat((cls_tokens, self.feedback), dim=0).unsqueeze(0)

            for blk in self.blocks_meta:
                reas_encoded = blk(x_q=reas_encoded, x_k=reas_encoded, x_v=reas_encoded)

            # self.feedback = self.feedback_norm.forward(reas_encoded[0, :].squeeze())
            self.feedback = reas_encoded[0, 0, :].squeeze().to(self.device)

            loss_weights = self.loss_weight_mlp.forward(self.feedback)
            self.feedback = self.feedback.unsqueeze(0).expand(batch_size * self.grid_size**2 * self.num_candidates, -1)

            # # for no skip connection, use this
            embed_reshaped = self.combiner(torch.cat([embed_reshaped, self.feedback], dim=-1))

            # for skip connection, use this
            # embed_reshaped = embed_reshaped + self.combiner(torch.cat([embed_reshaped, self.feedback], dim=-1))

        else:
            # set loss terms equal each time feedback is reset
            loss_weights = torch.ones(self.num_loss_terms, device=self.device)/self.num_loss_terms

        # # if combining prior to positional encodings, use this
        # if self.feedback is not None:
        #     self.feedback_old = self.feedback
        #     self.feedback = self.feedback.unsqueeze(0).expand(batch_size * self.grid_size**2 * self.num_candidates, -1)
        #     # # for skip connection use this
        #     # embed_reshaped = embed_reshaped + self.combiner(torch.cat([embed_reshaped, self.feedback], dim=-1))
        #     # for no skip connection use this
        #     embed_reshaped = self.combiner(torch.cat([embed_reshaped, self.feedback], dim=-1))


        # reshape for concatenating positional embeddings
        x_1 = embed_reshaped.view(batch_size, self.num_candidates, self.grid_size**2, -1)  # x is (B, 8, 9, self.embed_dim*2)
        embeddings = x_1.clone()

        # expand positional embeddings to fit batch (B, 8, 9, embed_dim)
        pos_embed_final = self.pos_embed.unsqueeze(0).expand(batch_size, self.num_candidates,
                                                             self.grid_size**2, -1)

        # concatenate positional embeddings
        x_1 = torch.cat([x_1, pos_embed_final], dim=-1)

        # if combining after positional encodings, use this
        # if self.feedback is not None:
        #     self.feedback_old = self.feedback
        #     x_1_reshaped = x_1.view(batch_size * self.num_candidates * self.grid_size ** 2, -1)
        #     self.feedback = self.feedback.expand(batch_size * self.grid_size**2 * self.num_candidates, -1)
        #     x_1_reshaped = self.combiner(torch.cat([x_1_reshaped, self.feedback], dim=-1))
        #     x_1 = x_1_reshaped.view(batch_size, self.num_candidates, self.grid_size**2, -1)
        #
        #     # add back positional embeddings
        #     x_1 = torch.cat([x_1, pos_embed_final], dim=-1)

        # logging.info("Positional encodings added.\n")

        x_1_reshaped = x_1.view(batch_size * self.num_candidates, self.grid_size**2, -1)

        # logging.info("Beginning ternary operation...\n")

        if self.ternary_num == 1:
            x_ternary = self.ternary_operation(x_1_reshaped)
        elif self.ternary_num == 2:
            x_ternary = self.ternary_hadamard(x_1_reshaped)
        else:
            x_ternary = self.ternary_mlp(x_1_reshaped)

        # logging.info("Ternary operation complete.\n")

        x_2 = x_ternary.view(batch_size, self.num_candidates, self.grid_size*2, -1)

        # apply temporal context normalization
        x_1 = self.tcn_1.forward(x_1)
        x_2 = self.tcn_2.forward(x_2)

        # logging.info("TCN complete.\n")

        # reshape x for batch processing
        x_1 = x_1.view(batch_size*self.num_candidates, self.grid_size**2, -1)
        x_2 = x_2.view(batch_size*self.num_candidates, self.grid_size*2, -1)

        # clone x for passing to transformer blocks
        y = x_1.clone()

        y_pos = pos_embed_final.reshape(batch_size*self.num_candidates, self.grid_size**2, self.embed_dim)

        # logging.info("Initializing symbols...\n")

        # repeat symbols along batch dimension
        symbols_1 = self.symbols_1.unsqueeze(0)
        symbols_1 = symbols_1.repeat(batch_size * self.num_candidates, 1, 1)
        symbols_2 = self.symbols_2.unsqueeze(0)
        symbols_2 = symbols_2.repeat(batch_size * self.num_candidates, 1, 1)

        # logging.info("Begin abstractor one...\n")

        # multi-headed self-attention blocks of abstractor
        for idx, blk in enumerate(self.blocks_abs_1):
            if idx == 0:
                x_1 = blk(x_q=x_1, x_k=x_1, x_v=symbols_1)
            else:
                x_1 = blk(x_q=x_1, x_k=x_1, x_v=x_1)

        x_1 = self.norm_x_1(x_1)

        # logging.info("End abstractor one.\n")

        # logging.info("Begin abstractor two...\n")

        # multi-headed self-attention blocks of abstractor
        for idx, blk in enumerate(self.blocks_abs_2):
            if idx == 0:
                x_2 = blk(x_q=x_2, x_k=x_2, x_v=symbols_2)
            else:
                x_2 = blk(x_q=x_2, x_k=x_2, x_v=x_2)

        x_2 = self.norm_x_2(x_2)

        # logging.info("End abstractor two.\n")

        # logging.info("Begin transformer...\n")

        # multi-headed self-attention blocks of transformer
        for idx, blk in enumerate(self.blocks_trans):
            y = blk(x_q=y_pos, x_k=y_pos, x_v=y) if self.restrict_qk else blk(x_q=y, x_k=y, x_v=y)

        y = self.norm_y(y)

        # logging.info("End transformer.\n")

        x_1 = x_1.view([batch_size, self.num_candidates, self.grid_size**2, -1])

        x_2 = x_2.view([batch_size, self.num_candidates, self.grid_size*2, -1])

        y = y.view(batch_size, self.num_candidates, self.grid_size**2, -1)
        y = self.tcn_1.inverse(y)

        # logging.info("Inverse TCN complete. Entering guesser head...\n")

        z = torch.cat([x_1, y], dim=-1)

        z_reshaped = torch.cat([z.mean(dim=-2), x_2.mean(dim=-2)], dim=-1).view(
            batch_size * self.num_candidates, -1)
        reas_raw = z_reshaped.clone() # (batch_size * num_candidates, -1)

        reas_encoded, reas_decoded = self.reas_autoencoder.forward(reas_raw.view(batch_size, self.num_candidates, -1))
        reas_decoded = reas_decoded.view(batch_size * self.num_candidates, -1)

        self.feedback = reas_encoded.clone().detach() # save tensor for feedback processing in next batch

        reas_encoded_expanded = reas_encoded.unsqueeze(1).expand(-1, self.num_candidates, -1).contiguous()

        # if incorporating meta-reasoning vector into guesser head, use this
        reas_meta_reas = torch.cat([z_reshaped,
                                    reas_encoded_expanded.view(batch_size*self.num_candidates, -1)], dim=-1)

        # # if not incorporating meta-reasoning vector into guesser head, use this
        # reas_meta_reas = z_reshaped

        dist_reshaped = self.guesser_head(reas_meta_reas)

        dist = dist_reshaped.view(batch_size, self.num_candidates) # for output

        # logging.info(f"self.feedback dimension: {self.feedback.size()}")

        # logging.info("Producing image recreation.\n")
        recreation = self.decoder.forward(embed_cached).view(batch_size, self.num_candidates,
                                                               self.grid_size**2, 1, 160, 160)

        # logging.info("Forward pass complete.\n")

        return dist, recreation, embeddings, reas_raw, reas_decoded, reas_meta_reas, loss_weights

    def encode(self, images):
        embeddings = self.perception.forward(images)  # takes input (B, 1, 160, 160), gives output (B, embed_dim)

        return embeddings

    def decode(self, embeddings):
        images = self.decoder.forward(embeddings)  # takes input (B, embed_dim), gives output (B, 1, 160, 160)

        return images

class L2Norm(nn.Module):
    def __init__(self, dim=None, eps=1e-12):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x / torch.norm(x, p=2, dim=self.dim, keepdim=True).clamp(min=self.eps)

class LossWeightingMLP(nn.Module):
    def __init__(self, feedback_dim, num_loss_terms, hidden_dim=128):
        super(LossWeightingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_loss_terms)
        )

    def forward(self, feedback_vector):
        # Produce weights for each loss term
        raw_output = self.mlp(feedback_vector)
        logging.info(f"Raw Output Before Softmax: {raw_output}")
        weights = F.softmax(raw_output, dim=-1)

        return weights

class AutoencoderBottleneck(nn.Module):
    def __init__(self, input_dim=512, bottleneck_dim=32):
        super(AutoencoderBottleneck, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoencoderBottleneckAlt(nn.Module):
    def __init__(self,
                 input_dim=1024,
                 bottleneck_dim=128,
                 depth=2,
                 num_heads=4,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm,
                 proj_drop=0.3,
                 attn_drop=0.3,
                 drop_path_max=0.5,
                 seq_length=8
                 ):
        super(AutoencoderBottleneckAlt, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length

        self.encoder = nn.Sequential(
            TransformerEncoder(depth, input_dim, num_heads, mlp_ratio, norm_layer, proj_drop, attn_drop, drop_path_max),
            nn.Flatten(),
            nn.Linear(input_dim * seq_length, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, bottleneck_dim),
            norm_layer(bottleneck_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            norm_layer(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim * seq_length),
            norm_layer(input_dim * seq_length),
            nn.Unflatten(1, (seq_length, input_dim)),
            TransformerDecoder(depth, input_dim, num_heads, mlp_ratio, norm_layer, proj_drop, attn_drop, drop_path_max)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio, norm_layer, proj_drop, attn_drop, drop_path_max):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, dim, num_heads, mlp_ratio,
                                           q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer,
                                           proj_drop=proj_drop,
                                           attn_drop=attn_drop, drop_path=drop_path_max * ((i + 1) / depth),
                                           restrict_qk=False) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, x, x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio, norm_layer, proj_drop, attn_drop, drop_path_max):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, dim, num_heads, mlp_ratio,
                                           q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer,
                                           proj_drop=proj_drop,
                                           attn_drop=attn_drop, drop_path=drop_path_max * ((i + 1) / depth),
                                           restrict_qk=False) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, x, x)
        return x


class DynamicWeightingRNN(nn.Module):
    def __init__(self,
                 input_dim=2,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.1,
                 output_dim=3):
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
                 output_dim=2):
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


# ResNet encoder
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
    def __init__(self, embed_dim, mlp_drop=0.5):
        super(ResNetEncoder, self).__init__()

        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(  # from N, 1, 160, 160
            ResidualBlock(1, 16),  # N, 16, 160, 160
            ResidualBlock(16, 32, 2),  # N, 32, 80, 80
            ResidualBlock(32, 64, 2),  # N, 64, 40, 40
            ResidualBlock(64, 128, 2),  # N, 128, 20, 20
            ResidualBlock(128, 256, 2),  # N, 256, 10, 10
            nn.Flatten(),  # N, 256*10*10
            nn.Linear(256*10*10, self.embed_dim),  # N, embed_dim
            nn.Dropout(p=mlp_drop)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class ResNetDecoder(nn.Module):
    def __init__(self, embed_dim=512, mlp_drop=0.5):
        super(ResNetDecoder, self).__init__()

        self.embed_dim = embed_dim

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 256*10*10),
            nn.Dropout(p=mlp_drop),
            nn.Unflatten(1, (256,10,10)),
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


class BackbonePerceptionAlt(nn.Module):
    def __init__(self,
                 embed_dim,
                 out_channels=256,
                 num_heads=32,
                 mlp_ratio=4,
                 norm_layer=nn.LayerNorm,
                 depth=4,
                 mlp_drop=0.3,
                 grid_size=10):
        super(BackbonePerceptionAlt, self).__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.grid_size = grid_size

        self.encoder = nn.Sequential( # from N, 1, 160, 160
            ResidualBlock(1, 16),  # N, 16, 160, 160
            ResidualBlock(16, 32, 2),  # N, 32, 80, 80
            ResidualBlock(32, 64, 2),  # N, 64, 40, 40
            ResidualBlock(64, 128, 2),  # N, 128, 20, 20
            ResidualBlock(128, 256, 2),  # N, 256, 10, 10
        )

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_size ** 2, self.out_channels]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.out_channels, grid_size=self.grid_size, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(self.out_channels, self.out_channels*2, self.num_heads, self.mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1,
                  drop_path=0.5*((i+1)/self.depth), restrict_qk=True) for i in range(self.depth)])

        self.mlp = nn.Linear(self.out_channels * 2 * self.grid_size**2, self.embed_dim)
        self.dropout = nn.Dropout(p=mlp_drop)

    def forward(self, x):

        batch_dim = x.size(0)

        x = self.encoder(x)

        x = x.reshape(batch_dim, self.grid_size**2, self.out_channels)

        # expand positional embeddings to fit batch (B, self.grid_size**2, embed_dim)
        pos_embed_final = self.pos_embed.unsqueeze(0).expand(batch_dim, self.grid_size ** 2, self.out_channels)

        # concatenate positional embeddings
        x = torch.cat([x, pos_embed_final], dim=-1)

        for block in self.blocks:
            x = block(x_q=pos_embed_final, x_k=pos_embed_final, x_v=x)

        x = x.reshape(batch_dim, self.out_channels * 2 * self.grid_size**2)

        x = self.dropout(self.mlp(x))

        return x


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

        # initialize and retrieve positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros([self.grid_dim ** 2, self.out_channels]), requires_grad=False)
        pos_embed = pos.get_2d_sincos_pos_embed(embed_dim=self.out_channels, grid_size=self.grid_dim, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        self.blocks = nn.ModuleList([
            Block(self.out_channels, self.out_channels, self.num_heads, self.mlp_ratio,
                  q_bias=False, k_bias=False, v_bias=False, norm_layer=norm_layer, proj_drop=0.3, attn_drop=0.3,
                  drop_path=0.5*((i+1)/self.depth), restrict_qk=False) for i in range(self.depth)])

        self.mlp = nn.Linear(self.out_channels * self.grid_dim**2, self.embed_dim)
        self.dropout = nn.Dropout(p=mlp_drop)

    def forward(self, x):

        batch_dim = x.size(0)

        # logging.info("Begin encoder call...\n")

        x = self.encoder(x)

        # logging.info("End encoder call.\n")

        x = x.reshape(batch_dim, self.grid_dim ** 2, self.out_channels)

        # expand positional embeddings to fit batch (B, self.grid_dim**2, embed_dim)
        # pos_embed_final = self.pos_embed.unsqueeze(0).expand(batch_dim, self.grid_dim ** 2, self.out_channels)

        # add positional embeddings
        # x = x + pos_embed_final

        # logging.info("Positional encodings (not) added.\n")

        # logging.info("Begin transformer...\n")

        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        # logging.info("End transformer...\n")

        x = x.reshape(batch_dim, self.out_channels * self.grid_dim**2)

        # logging.info("Begin MLP...\n")

        x = self.dropout(self.mlp(x))

        # logging.info("End MLP...\n")

        return x


class BackboneDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=32, mlp_ratio=4, norm_layer=nn.LayerNorm, depth=4, mlp_drop=0.5):
        super(BackboneDecoder, self).__init__()

        self.embed_dim = embed_dim

        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.mlp_to_tr = nn.Sequential(nn.Linear(self.embed_dim, 256 * 10 * 10),
                                       nn.Dropout(p=mlp_drop),
                                       nn.Unflatten(1, (100, 256)))

        self.blocks = nn.ModuleList([
            Block(256, 256, self.num_heads, self.mlp_ratio,
                  q_bias=True, k_bias=True, v_bias=True, norm_layer=norm_layer, proj_drop=0.1, attn_drop=0.1,
                  drop_path=0.5 * ((i + 1) / self.depth)) for i in range(self.depth)])

        self.decoder = nn.Sequential(
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

        batch_dim = x.size(0)

        # pass through MLP and reshape
        x = self.mlp_to_tr(x)

        # pass through transformer
        for block in self.blocks:
            x = block(x_q=x, x_k=x, x_v=x)

        # transpose and reshape
        x = x.transpose(1, 2).reshape(batch_dim, 256, 10, 10)

        # apply deconvolution
        x = self.decoder(x)

        return x


class MLPDecoder(nn.Module):
    def __init__(self, embed_dim=512, mlp_drop=0.5):
        super(MLPDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 80 * 80),
            # nn.Dropout(p=mlp_drop//2),
            nn.ReLU(),
            nn.Linear(80 * 80, 160 * 160),
            nn.Dropout(p=mlp_drop),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = x.view(-1, 1, 160, 160)  # Reshape the output to the desired dimensions
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

