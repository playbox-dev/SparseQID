# Portions Copyright (c) Ruopeng Gao. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for SparseQID.

"""Build SparseQID identity targets for a training clip.

Adapted from MOTIP's ``GenerateIDLabels`` and
``TurnIntoTrajectoryAndUnknown`` transforms. Converts per-frame ground-truth
track IDs into relative-ID labels, detection-index maps, padding masks, time
indices, occlusion and switch augmentation, and newborn labels.
"""

from __future__ import annotations

import math
import random

import einops
import torch
from torch import nn


def pos3d_encoding(xyz, dim=256, scale=10.0):
    """Fourier encoding of 3D centers (the DETR->3D swap): (..., 3) -> (..., dim)."""
    d = max(dim // 6, 1)
    freqs = scale ** (torch.arange(d, device=xyz.device, dtype=torch.float32) / max(d - 1, 1))
    a = xyz[..., None] * freqs  # (..., 3, d)
    enc = torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)  # (..., 6d)
    if enc.shape[-1] < dim:
        enc = torch.cat([enc, enc.new_zeros(*enc.shape[:-1], dim - enc.shape[-1])], -1)
    return enc[..., :dim]


class PosEncoder(nn.Module):
    """3D-centre -> dim position encoding, switchable for the encoding-design ablation.
    Computes in float32 (matches pos3d_encoding) so it's autocast-safe when the result
    is added to the appearance embeddings.

      - "fourier": the deployed multi-frequency sin/cos (PARAMETERLESS -> state_dict is
        empty, so a fourier run is byte-identical to the pre-ablation code path).
      - "raw":     a single learnable Linear(3, dim) -- position injected, but no
        multi-frequency structure (isolates whether Fourier specifically matters).
      - "mlp":     Linear(3, dim) -> ReLU -> Linear(dim, dim), a nonlinear learnable
        encoder (the other "any position injection" baseline).
    """

    def __init__(self, mode="fourier", dim=256, scale=10.0):
        super().__init__()
        assert mode in ("fourier", "raw", "mlp"), mode
        self.mode, self.dim, self.scale = mode, dim, scale
        if mode == "raw":
            self.net = nn.Linear(3, dim)
        elif mode == "mlp":
            self.net = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        else:
            self.net = None  # fourier: no params

    def forward(self, xyz):
        # xyz: (..., 3), possibly 0 rows. Keep float32 for a stable add to embeddings.
        xyz = xyz.float()
        if self.mode == "fourier":
            return pos3d_encoding(xyz, self.dim, self.scale)
        return self.net(xyz)


def build_id_targets(
    frame_ids,
    num_id_vocabulary,
    aug_num_groups,
    num_training_ids,
    occlusion_prob,
    switch_prob,
    rand_id_prob=1.0,
):
    """frame_ids: list (len T) of 1-D LongTensors -- the GT track id of each matched
    detection in that frame (the i-th entry's value is the global track id; its index
    i is the detection's ann index in that frame). Returns a dict of (G,T,N) tensors."""
    T = len(frame_ids)
    G = aug_num_groups
    ids_set = set()
    for ids in frame_ids:
        ids_set.update(ids.tolist())
    ids_list = sorted(ids_set)
    N = len(ids_list)
    id_to_idx = {tid: n for n, tid in enumerate(ids_list)}

    # base (T,N): mask True = track absent that frame; ann_idx = detection index
    base_masks = torch.ones((T, N), dtype=torch.bool)
    base_ann = -torch.ones((T, N), dtype=torch.int64)
    for t, ids in enumerate(frame_ids):
        for i, tid in enumerate(ids.tolist()):
            n = id_to_idx[tid]
            base_masks[t, n] = False
            base_ann[t, n] = i

    # cap to vocab / training budget
    cap = num_training_ids if N > num_training_ids else num_id_vocabulary
    if N > num_id_vocabulary or N > num_training_ids:
        sel = torch.randperm(N)[:cap]
        base_masks, base_ann, N = base_masks[:, sel], base_ann[:, sel], cap

    # per-group RANDOM relative-id assignment (the id augmentation)
    id_labels = torch.zeros((G, T, N), dtype=torch.int64)
    masks = torch.ones((G, T, N), dtype=torch.bool)
    ann = -torch.ones((G, T, N), dtype=torch.int64)
    for g in range(G):
        # rand_id_prob: per-group chance of the RANDOM relative-id assignment (the true
        # in-context copy objective). Otherwise a deterministic sorted-order mapping —
        # a learnable anchor that eases the copy circuit in (curriculum).
        rid = (
            torch.randperm(num_id_vocabulary)[:N]
            if random.random() < rand_id_prob
            else torch.arange(N)
        )[None].repeat(T, 1)
        id_labels[g] = rid
        masks[g] = base_masks.clone()
        ann[g] = base_ann.clone()
    times = torch.arange(T, dtype=torch.int64)[None, :, None].repeat(G, 1, N)

    traj_id, traj_mask, traj_ann = id_labels.clone(), masks.clone(), ann.clone()
    unk_id, unk_mask, unk_ann = id_labels.clone(), masks.clone(), ann.clone()

    # --- trajectory occlusion: drop a random contiguous span per (group, track),
    #     applied to BOTH the trajectory and unknown sides so the object is fully
    #     invisible over the span (matches the original MOTIP transforms.py:500-506). ---
    if occlusion_prob > 0.0:
        tm = einops.rearrange(traj_mask, "G T N -> (G N) T")
        um = einops.rearrange(unk_mask, "G T N -> (G N) T")
        for i in range(G * N):
            if random.random() < occlusion_prob:
                b = random.randint(0, T - 1)
                e = b + math.ceil((T - 1 - b) * random.random())
                tm[i, b:e] = True
                um[i, b:e] = True
        traj_mask = einops.rearrange(tm, "(G N) T -> G T N", G=G, N=N)
        unk_mask = einops.rearrange(um, "(G N) T -> G T N", G=G, N=N)

    # --- trajectory switch: randomly swap matched dets across tracks per frame ---
    if switch_prob > 0.0:
        tl = einops.rearrange(traj_id, "G T N -> (G T) N")
        tmk = einops.rearrange(traj_mask, "G T N -> (G T) N")
        ta = einops.rearrange(traj_ann, "G T N -> (G T) N")
        for gt in range(G * T):
            sw = torch.nonzero(torch.bernoulli(torch.full((N,), switch_prob)))[:, 0]
            if len(sw) > 1:
                perm = sw[torch.randperm(len(sw))]
                ta[gt, sw] = ta[gt, perm]
                tmk[gt, sw] = tmk[gt, perm]
        traj_id = einops.rearrange(tl, "(G T) N -> G T N", G=G, T=T)
        traj_mask = einops.rearrange(tmk, "(G T) N -> G T N", G=G, T=T)
        traj_ann = einops.rearrange(ta, "(G T) N -> G T N", G=G, T=T)

    # --- newborn label on the unknown side: a track's first appearance = newborn ---
    ul = einops.rearrange(unk_id, "G T N -> (G N) T")
    tmk = einops.rearrange(traj_mask, "G T N -> (G N) T")
    born = torch.cumsum(~tmk, dim=1) > 0
    newborn = ~torch.cat([torch.zeros((G * N, 1), dtype=torch.bool), born[:, :-1]], dim=-1)
    ul[newborn] = num_id_vocabulary
    unk_id = einops.rearrange(ul, "(G N) T -> G T N", G=G, N=N)

    return {
        "trajectory_id_labels": traj_id,
        "trajectory_masks": traj_mask,
        "trajectory_ann_idxs": traj_ann,
        "trajectory_times": times.clone(),
        "unknown_id_labels": unk_id,
        "unknown_masks": unk_mask,
        "unknown_ann_idxs": unk_ann,
        "unknown_times": times.clone(),
    }


def assemble_seq_info(targets, frame_embeds, pos_enc=None, pos_only=False):
    """Fill (B=1,G,T,N,C) trajectory/unknown feature tensors from per-frame detection
    embeddings, using the ann-index map. frame_embeds: list (len T) of (D_t, C) matched
    detection embeddings (row i = the i-th matched detection in frame t). pos_enc: optional
    list (len T) of (D_t, C) 3D positional encodings added to the embeddings (the 3D swap)."""
    G, T, N = targets["trajectory_id_labels"].shape
    C = (
        frame_embeds[0].shape[-1]
        if len(frame_embeds[0])
        else next(e.shape[-1] for e in frame_embeds if len(e))
    )
    dev = (
        frame_embeds[
            next(t for t in range(T) if len(frame_embeds[t]))
            if any(len(e) for e in frame_embeds)
            else 0
        ].device
        if any(len(e) for e in frame_embeds)
        else torch.device("cpu")
    )

    def gather(ann):
        feat = torch.zeros((G, T, N, C), device=dev)
        for t in range(T):
            e = frame_embeds[t]
            if pos_enc is not None and len(e):
                e = pos_enc[t] if pos_only else e + pos_enc[t]  # pos_only: drop appearance
            for g in range(G):
                idx = ann[g, t]  # (N,) detection index per slot (-1 = none)
                ok = idx >= 0
                if ok.any() and len(e):
                    feat[g, t, ok] = e[idx[ok]]
        return feat[None]  # add batch dim B=1

    seq = {k: v[None].to(dev) for k, v in targets.items()}  # add B dim
    seq["trajectory_features"] = gather(targets["trajectory_ann_idxs"])
    seq["unknown_features"] = gather(targets["unknown_ann_idxs"])
    return seq
