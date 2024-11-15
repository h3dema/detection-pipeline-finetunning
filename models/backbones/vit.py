import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.layers import (
    Backbone,
    PatchEmbed,
    Block,
    get_abs_pos,
    get_norm,
    Conv2d,
    LastLevelMaxPool
)


class ViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
    ):
        """

        :param img_size (int): Input image size.
        :param patch_size (int): Patch size.
        :param in_chans (int): Number of input image channels.
        :param embed_dim (int): Patch embedding dimension.
        :param depth (int): Depth of ViT.
        :param num_heads (int): Number of attention heads in each ViT block.
        :param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        :param qkv_bias (bool): If True, add a learnable bias to query, key, value.
        :param drop_path_rate (float): Stochastic depth rate.
        :param norm_layer (nn.Module): Normalization layer.
        :param act_layer (nn.Module): Activation layer.
        :param use_abs_pos (bool): If True, use absolute positional embeddings.
        :param use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
        :param rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
        :param window_size (int): Window size for window attention blocks.
        :param window_block_indexes (list): Indexes for blocks using window attention.
        :param residual_block_indexes (list): Indexes for blocks using conv propagation.
        :param use_act_checkpoint (bool): If True, use activation checkpointing.
        :param pretrain_img_size (int): input image size for pretraining models.
        :param pretrain_use_cls_token (bool): If True, pretrainig models use class token.
        :param out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights of the model.

        This function is taken from the source code of the `vitdet` model.
        https://github.com/IDEA-Research/DAB-DETR/blob/main/models/vitdet.py
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor of shape (N,C,H,W)
        :return: A dictionary with a single key-value pair.
                 The key is the feature name and the value is the feature map tensor.
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )

        for blk in self.blocks:
            x = blk(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs
