from __future__ import annotations

import torch

from sparseqid.identity import QueryMatcher


def test_query_matcher_assigns_identity_labels_by_class_and_box_cost() -> None:
    matcher = QueryMatcher(reg_weights=[1.0] * 11)
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [-8.0, -8.0]]])
    boxes = torch.zeros((1, 3, 11))
    boxes[..., 7] = 1.0
    ground_truth = torch.tensor(
        [
            [2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    boxes[0, 1, 0] = 2.0

    result = matcher(
        logits,
        boxes,
        [torch.tensor([1, 0])],
        [ground_truth],
        [torch.tensor([101, 202])],
    )

    assert result.tolist() == [[202, 101, -1]]
