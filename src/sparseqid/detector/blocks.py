# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model building blocks for Sparse4D."""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn

import logging


def linear_relu_ln(
    embed_dims: int, in_loops: int, out_loops: int, input_dims: Optional[int] = None
):
    """Create a sequence of linear layers with ReLU and LayerNorm.

    Args:
        embed_dims (int): Embedding dimension
        in_loops (int): Number of inner linear-ReLU loops
        out_loops (int): Number of outer loops (each ending with LayerNorm)
        input_dims (int): Input dimensions (defaults to embed_dims)

    Returns:
        A list of PyTorch modules
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class DeformableFeatureAggregation(nn.Module):
    """Module for deformable feature aggregation from multi-view features."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        max_num_cams: int = 20,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: Any = None,
        use_camera_embed: bool = False,
        residual_mode: str = "add",
    ):
        """Initialize DeformableFeatureAggregation.

        Args:
            embed_dims (int): Embedding dimension
            num_groups (int): Number of attention groups
            num_levels (int): Number of feature levels
            num_cams (int): Number of cameras
            max_num_cams (int): Maximum number of cameras
            proj_drop (float): Dropout probability for projection
            attn_drop (float): Dropout probability for attention
            kps_generator (Any): Keypoints generator
            use_camera_embed (bool): Whether to use camera embeddings
            residual_mode (str): Residual connection mode ('add' or 'cat')
        """
        super(DeformableFeatureAggregation, self).__init__()

        # Validate configuration
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, but got {embed_dims} and {num_groups}"
            )

        # Initialize dimensions and configuration
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams

        # Initialize other parameters
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.feat_dims = self.embed_dims

        # Initialize projection dropout
        self.proj_drop = nn.Dropout(proj_drop)

        self.kps_generator = kps_generator
        self.num_pts = self.kps_generator.num_pts

        # Build projection layers
        self.anchor_embed_proj = nn.Identity()
        self.output_proj = nn.Linear(embed_dims, self.feat_dims)

        # Build camera encoder if needed
        if use_camera_embed:
            self.camera_encoder = nn.Sequential(*linear_relu_ln(self.feat_dims, 1, 2, 12))
            self.weights_fc = nn.Linear(self.feat_dims, num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                self.feat_dims, num_groups * max_num_cams * num_levels * self.num_pts
            )

    def init_weights(self):
        """Initialize model weights."""
        # Initialize weights_fc with zeros
        nn.init.zeros_(self.weights_fc.weight)
        if self.weights_fc.bias is not None:
            nn.init.zeros_(self.weights_fc.bias)

        # Initialize anchor_embed_proj with Xavier uniform
        if isinstance(self.anchor_embed_proj, nn.Linear):
            nn.init.xavier_uniform_(self.anchor_embed_proj.weight)
            if self.anchor_embed_proj.bias is not None:
                nn.init.zeros_(self.anchor_embed_proj.bias)

        # Initialize output_proj with Xavier uniform
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """Forward pass for deformable feature aggregation.

        Args:
            instance_feature (torch.Tensor): Instance features [B, N, D]
            anchor (torch.Tensor): Anchors [B, N, A]
            anchor_embed (torch.Tensor): Anchor embeddings [B, N, E]
            feature_maps (List[torch.Tensor]): List of feature maps
            metas (dict): Metadata dictionary
            kwargs (dict): Additional arguments

        Returns:
            Dictionary with aggregated features
        """
        bs, num_anchor = instance_feature.shape[:2]

        # Generate keypoints from anchors
        key_points = self.kps_generator(anchor, instance_feature)

        # Get number of cameras from projection matrices
        num_cams = metas["projection_mat"].shape[1]

        # Get attention weights
        weights = self._get_weights(instance_feature, anchor_embed, metas, num_cams)

        features = self.feature_sampling(
            feature_maps,
            key_points,
            metas["projection_mat"],
            metas["image_wh"],
        )
        features = self.multi_view_level_fusion(features, weights)
        features = features.sum(dim=2)

        # Apply projection and dropout
        output = self.proj_drop(self.output_proj(features))

        # Apply residual connection
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return {"instance_feature": output}

    def _get_weights(self, instance_feature, anchor_embed, metas=None, num_cams=None):
        """Get attention weights for feature aggregation.

        Args:
            instance_feature (torch.Tensor): Instance features
            anchor_embed (torch.Tensor): Anchor embeddings
            metas (dict): Metadata dictionary
            num_cams (int): Number of cameras

        Returns:
            Attention weights
        """
        bs, num_anchor = instance_feature.shape[:2]

        # Project anchor embeddings
        anchor_embed = self.anchor_embed_proj(anchor_embed)

        # Combine instance features and anchor embeddings
        feature = instance_feature + anchor_embed

        # Use default num_cams if not specified
        if num_cams is None:
            num_cams = self.num_cams

        # Apply camera embeddings if needed
        if self.camera_encoder is not None:
            # Extract camera parameters
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, num_cams, -1)
            )

            # Combine with instance features
            feature = feature[:, :, None] + camera_embed[:, None]
            weights = self.weights_fc(feature)
        else:
            # Generate weights directly
            weights = self.weights_fc(feature)

            # Reshape and slice the output based on the current num_cams
            weights = weights[:, :, : (self.num_groups * num_cams * self.num_levels * self.num_pts)]

        weights = (
            weights.reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(bs, num_anchor, num_cams, self.num_levels, self.num_pts, self.num_groups)
        )

        # Apply dropout during training
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)

        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """Project 3D keypoints to 2D image coordinates.

        Args:
            key_points (torch.Tensor): 3D keypoints [B, N, P, 3]
            projection_mat (torch.Tensor): Projection matrices [B, C, 4, 4]
            image_wh (torch.Tensor): Image dimensions (optional)

        Returns:
            2D points in image coordinates
        """
        # Add homogeneous coordinate (1) to keypoints
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)

        # Apply projection matrix
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)

        # Convert to image coordinates
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)

        # Normalize by image dimensions if provided
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]

        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample features at projected 2D points.

        Args:
            feature_maps (List[torch.Tensor]): List of feature maps [B, C, H, W]
            key_points (torch.Tensor): 3D keypoints [B, N, P, 3]
            projection_mat (torch.Tensor): Projection matrices [B, C, 4, 4]
            image_wh (torch.Tensor): Image dimensions (optional)

        Returns:
            Sampled features [B, N, C, L, P, D]
        """
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Project keypoints to 2D
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )

        # Convert to grid_sample format [-1, 1]
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        # Sample features using grid_sample
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d, align_corners=False
                )
            )

        # Stack and reshape features
        features = torch.stack(features, dim=1)
        features = features.reshape(bs, num_cams, num_levels, -1, num_anchor, num_pts).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Fuse features across views and levels.

        Args:
            features (torch.Tensor): Sampled features [B, N, C, L, P, D]
            weights (torch.Tensor): Attention weights [B, N, C, L, P, G]

        Returns:
            Fused features [B, N, P, D]
        """
        bs, num_anchor = weights.shape[:2]

        # Reshape features for grouped attention
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )

        # Sum across views and levels
        features = features.sum(dim=2).sum(dim=2)

        # Reshape to final output size
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)

        return features

class AsymmetricFFN(nn.Module):
    """Asymmetric Feed-Forward Network with optional pre-norm."""

    def __init__(
        self,
        in_channels: Optional[int] = None,
        pre_norm: Optional[Dict[str, Any]] = None,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: Dict[str, Any] = {"type": "ReLU", "inplace": True},
        ffn_drop: float = 0.0,
        dropout_layer: Optional[Dict[str, Any]] = None,
        add_identity: bool = True,
        **kwargs,
    ):
        """Initialize AsymmetricFFN.

        Args:
            in_channels (int): Input channel dimension
            pre_norm (dict): Pre-normalization configuration
            embed_dims (int): Output embedding dimension
            feedforward_channels (int): Hidden dimension of feedforward network
            num_fcs (int): Number of fully-connected layers
            act_cfg (dict): Activation configuration
            ffn_drop (float): Dropout probability
            dropout_layer (dict): Dropout layer configuration
            add_identity (bool): Whether to add identity shortcut
        """
        super(AsymmetricFFN, self).__init__()

        # Validate configuration
        assert num_fcs >= 2, f"num_fcs should be no less than 2. got {num_fcs}."

        # Initialize parameters
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg

        # Create activation layer
        if act_cfg["type"] == "ReLU":
            self.activate = nn.ReLU(inplace=act_cfg["inplace"])
        elif act_cfg["type"] == "GELU":
            self.activate = nn.GELU()
        elif act_cfg["type"] == "LeakyReLU":
            self.activate = nn.LeakyReLU(
                negative_slope=act_cfg["negative_slope"], inplace=act_cfg["inplace"]
            )
        else:
            raise ValueError(f"Unsupported activation type: {act_cfg['type']}")

        # Build FFN layers
        layers = []

        # Use input channels if provided, otherwise use embed_dims
        if in_channels is None:
            in_channels = embed_dims

        # Build pre-norm layer if provided
        if pre_norm is not None:
            if pre_norm["type"] == "LN":
                self.pre_norm = nn.LayerNorm(in_channels)
            elif pre_norm["type"] == "BN":
                self.pre_norm = nn.BatchNorm1d(in_channels)
            else:
                raise ValueError(f"Unsupported norm type: {pre_norm['type']}")
        else:
            self.pre_norm = None

        # Build intermediate layers
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels

        # Build output layer
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))

        # Create sequential module for all layers
        self.layers = nn.Sequential(*layers)

        # Build dropout layer
        if dropout_layer is not None:
            if dropout_layer["type"] == "Dropout":
                self.dropout_layer = nn.Dropout(dropout_layer["drop_prob"])
            elif dropout_layer["type"] == "DropPath":
                from timm.models.layers import DropPath

                self.dropout_layer = DropPath(dropout_layer["drop_prob"])
            else:
                raise ValueError(f"Unsupported dropout type: {dropout_layer['type']}")
        else:
            self.dropout_layer = nn.Identity()

        # Setup identity connection
        self.add_identity = add_identity
        if self.add_identity:
            if self.in_channels == embed_dims:
                self.identity_fc = nn.Identity()
            else:
                self.identity_fc = nn.Linear(self.in_channels, embed_dims)

    def forward(self, x, identity=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor
            identity (torch.Tensor): Identity tensor (optional)

        Returns:
            Output tensor
        """
        # Apply pre-normalization if available
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # Pass through FFN layers
        out = self.layers(x)

        # Return without identity if not needed
        if not self.add_identity:
            return self.dropout_layer(out)

        # Use input as identity if not provided
        if identity is None:
            identity = x

        # Apply identity projection
        identity = self.identity_fc(identity)

        # Add identity connection with dropout
        return identity + self.dropout_layer(out)


class Dropout(nn.Dropout):
    """A wrapper for ``torch.nn.Dropout``, We rename the ``p`` of
    ``torch.nn.Dropout`` to ``drop_prob`` so as to be consistent with
    ``DropPath``

    Args:
        drop_prob (float): Probability of the elements to be
            zeroed. Default: 0.5.
        inplace (bool):  Do the operation inplace or not. Default: False.
    """

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


class MultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__()
        if "dropout" in kwargs:
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        if dropout_layer is not None:
            self.dropout_layer = nn.Dropout(dropout_layer["drop_prob"])
        else:
            self.dropout_layer = nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    logging.info(
                        f"position encoding of key is missing in {self.__class__.__name__}."
                    )
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))
