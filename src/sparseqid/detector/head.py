# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse4D Head for Sparse4D."""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict, Any

from .blocks import DeformableFeatureAggregation
from .instance_bank import InstanceBank
from .detection3d_blocks import SparseBox3DEncoder
from .detection3d_blocks import SparseBox3DKeyPointsGenerator
from .detection3d_blocks import SparseBox3DRefinementModule
from .blocks import AsymmetricFFN
from .blocks import MultiheadAttention


class Sparse4DHead(nn.Module):
    """Sparse4D head."""

    def __init__(
        self,
        config: Dict[str, Any],
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        operation_order: Optional[List[str]] = None,
        decouple_attn: bool = True,
        return_feature: bool = False,
    ):
        """Initialize Sparse4DHead.

        Args:
            instance_bank (InstanceBank): Instance bank.
            anchor_encoder (AnchorEncoder): Anchor encoder.
            graph_model (nn.Module): Graph model.
            norm_layer (nn.Module): Normalization layer.
            ffn (nn.Module): Feed-forward network.
            deformable_model (nn.Module): Deformable model.
            refine_layer (nn.Module): Refinement layer.
            num_decoder (int): Number of decoder layers.
            num_single_frame_decoder (int): Number of single frame decoder layers.
            temp_graph_model (nn.Module): Temporal graph model.
            operation_order (List): Order of operations.
            decouple_attn (bool): Whether to decouple attention.
            return_feature (bool): Whether to return feature.
        """
        super(Sparse4DHead, self).__init__()

        self.config = config
        self.head_cfg = config["model"]["head"]
        self.instance_bank_cfg = self.head_cfg["instance_bank"]
        self.anchor_encoder_cfg = self.head_cfg["anchor_encoder"]
        self.graph_model_cfg = self.head_cfg["graph_model"]
        self.ffn_cfg = self.head_cfg["ffn"]
        self.deformable_cfg = self.head_cfg["deformable_model"]
        self.refine_cfg = self.head_cfg["refine_layer"]
        self.temp_graph_cfg = self.head_cfg["temp_graph_model"]
        self.score_threshold = self.head_cfg["score_threshold"]
        self.num_single_frame_decoder = num_single_frame_decoder
        self.decouple_attn = decouple_attn
        self.return_feature = return_feature

        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        # =========== build modules ===========
        self.instance_bank = InstanceBank(
            num_anchor=self.instance_bank_cfg["num_anchor"],
            embed_dims=self.instance_bank_cfg["embed_dims"],
            anchor=self.instance_bank_cfg["anchor"],
            anchor_handler=SparseBox3DKeyPointsGenerator(),
            num_temp_instances=self.instance_bank_cfg["num_temp_instances"],
            default_time_interval=self.instance_bank_cfg["default_time_interval"],
            confidence_decay=self.instance_bank_cfg["confidence_decay"],
            feat_grad=self.instance_bank_cfg["feat_grad"],
        )
        self.anchor_encoder = SparseBox3DEncoder(
            embed_dims=self.anchor_encoder_cfg["embed_dims"],
            vel_dims=self.anchor_encoder_cfg["vel_dims"],
            mode=self.anchor_encoder_cfg["mode"],
            output_fc=self.anchor_encoder_cfg["output_fc"],
            in_loops=self.anchor_encoder_cfg["in_loops"],
            out_loops=self.anchor_encoder_cfg["out_loops"],
        )
        # Create layers based on operation order
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op == "temp_gnn":
                self.layers.append(
                    MultiheadAttention(
                        embed_dims=self.temp_graph_cfg["embed_dims"],
                        num_heads=self.temp_graph_cfg["num_heads"],
                        dropout=self.temp_graph_cfg["dropout"],
                        batch_first=self.temp_graph_cfg["batch_first"],
                    )
                )
            elif op == "gnn":
                self.layers.append(
                    MultiheadAttention(
                        embed_dims=self.graph_model_cfg["embed_dims"],
                        num_heads=self.graph_model_cfg["num_heads"],
                        dropout=self.graph_model_cfg["dropout"],
                        batch_first=self.graph_model_cfg["batch_first"],
                    )
                )
            elif op == "norm":
                self.layers.append(
                    nn.LayerNorm(
                        normalized_shape=256,
                        eps=1e-5,
                    )
                )
            elif op == "ffn":
                self.layers.append(
                    AsymmetricFFN(
                        in_channels=self.ffn_cfg["in_channels"],
                        pre_norm=self.ffn_cfg["pre_norm"],
                        embed_dims=self.ffn_cfg["embed_dims"],
                        feedforward_channels=self.ffn_cfg["feedforward_channels"],
                        num_fcs=self.ffn_cfg["num_fcs"],
                        act_cfg=self.ffn_cfg["act_cfg"],
                        ffn_drop=self.ffn_cfg["ffn_drop"],
                    )
                )
            elif op == "deformable":
                self.layers.append(
                    DeformableFeatureAggregation(
                        embed_dims=self.deformable_cfg["embed_dims"],
                        num_groups=self.deformable_cfg["num_groups"],
                        num_levels=self.deformable_cfg["num_levels"],
                        num_cams=self.deformable_cfg["num_cams"],
                        max_num_cams=self.deformable_cfg["max_num_cams"],
                        proj_drop=self.deformable_cfg["proj_drop"],
                        attn_drop=self.deformable_cfg["attn_drop"],
                        kps_generator=SparseBox3DKeyPointsGenerator(
                            embed_dims=self.deformable_cfg["kps_generator"]["embed_dims"],
                            num_learnable_pts=self.deformable_cfg["kps_generator"][
                                "num_learnable_pts"
                            ],
                            fix_scale=self.deformable_cfg["kps_generator"]["fix_scale"],
                        ),
                        use_camera_embed=self.deformable_cfg["use_camera_embed"],
                        residual_mode=self.deformable_cfg["residual_mode"],
                    )
                )
            elif op == "refine":
                self.layers.append(
                    SparseBox3DRefinementModule(
                        embed_dims=self.refine_cfg["embed_dims"],
                        num_cls=len(self.config["dataset"]["classes"]),
                        refine_yaw=self.refine_cfg["refine_yaw"],
                        with_quality_estimation=self.refine_cfg["with_quality_estimation"],
                    )
                )
            else:
                self.layers.append(None)

        self.embed_dims = self.instance_bank.embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.init_weights()

    def init_weights(self):
        """Initialize the weights of the model."""
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        """Forward function."""
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        """Run the recurrent detector head for one synchronized frame."""
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(batch_size, metas)

        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = (
            self.anchor_encoder(temp_anchor) if temp_anchor is not None else None
        )

        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )["instance_feature"]

            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {
            "classification": classification,
            "prediction": prediction,
            "quality": quality,
        }
        self.instance_bank.cache(instance_feature, anchor, cls, metas, feature_maps)
        if not self.training:
            output["instance_id"] = self.instance_bank.get_instance_id(
                cls, anchor, self.score_threshold
            )
        if self.return_feature:
            output["instance_feature"] = instance_feature
        return output

def build_head(config: Dict[str, Any]) -> nn.Module:
    """Build a detection head according to the configuration.

    Args:
        config: Configuration dictionary for the head

    Returns:
        nn.Module: Detection head
    """
    head_config = config["model"]["head"]
    head_type = head_config["type"]

    if head_type == "sparse4d":
        # Create head with the constructed modules
        head = Sparse4DHead(
            config=config,
            num_decoder=head_config["num_decoder"],
            num_single_frame_decoder=head_config["num_single_frame_decoder"],
            operation_order=head_config["operation_order"],
            decouple_attn=head_config["decouple_attn"],
            return_feature=head_config["return_feature"],
        )

        return head
    else:
        raise ValueError(f"Unsupported head type: {head_type}")
