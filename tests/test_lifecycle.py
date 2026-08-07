"""Deterministic tests for SparseQID's online identity lifecycle."""

import torch

from sparseqid.tracking import SparseQIDTracker, track_clip

FEATURE_DIM = 8


class StubTrajectoryModeling:
    def eval(self):
        return self

    def __call__(self, sequence):
        return sequence


class StubDecoder:
    feature_dim = FEATURE_DIM

    def __init__(self, relative_position_length, vocabulary_size):
        self.rel_pe_length = relative_position_length
        self.vocabulary_size = vocabulary_size

    def eval(self):
        return self

    def __call__(self, sequence, use_decoder_checkpoint=False):
        del use_decoder_checkpoint
        trajectory_features = sequence["trajectory_features"][0, 0]
        trajectory_masks = sequence["trajectory_masks"][0, 0]
        trajectory_ids = sequence["trajectory_id_labels"][0, 0]
        unknown_features = sequence["unknown_features"][0, 0, 0]

        logits = torch.full((len(unknown_features), self.vocabulary_size + 1), -10.0)
        logits[:, self.vocabulary_size] = 5.0
        for column in range(trajectory_features.shape[1]):
            observations = (~trajectory_masks[:, column]).nonzero().flatten()
            if not len(observations):
                continue
            feature = trajectory_features[observations[-1], column]
            similarities = torch.nn.functional.cosine_similarity(
                unknown_features, feature[None], dim=-1
            )
            label = int(trajectory_ids[0, column])
            matches = similarities > 0.95
            logits[matches, label] = 20.0 * similarities[matches]
        return logits[None, None, None], None, None


def track(frames, vocabulary_size=4, miss_tolerance=30, relative_position_length=8):
    decoder = StubDecoder(relative_position_length, vocabulary_size)
    trajectory_modeling = StubTrajectoryModeling()
    embeddings = [torch.stack(frame) if frame else torch.zeros(0, FEATURE_DIM) for frame in frames]
    return track_clip(
        decoder,
        trajectory_modeling,
        embeddings,
        vocabulary_size,
        "cpu",
        miss_tolerance=miss_tolerance,
        identity_threshold=0.2,
    )


A, B, D = (torch.eye(FEATURE_DIM)[index] for index in range(3))


def test_clip_helper_uses_the_public_streaming_runtime():
    decoder = StubDecoder(relative_position_length=8, vocabulary_size=4)
    tracker = SparseQIDTracker(decoder, StubTrajectoryModeling(), 4, "cpu")
    frames = [[A], [A], [], [A]]
    streamed = [
        tracker.step(torch.stack(frame) if frame else torch.zeros(0, FEATURE_DIM))
        for frame in frames
    ]
    batched = track(frames)
    assert [ids.tolist() for ids in streamed] == [ids.tolist() for ids in batched]


def test_identity_persists_while_observed():
    global_ids = track([[A]] * 6)
    assert [int(ids[0]) for ids in global_ids] == [0] * 6


def test_identity_survives_gap_inside_memory_window():
    global_ids = track([[A], [A], [], [], [], [], [A]])
    assert int(global_ids[0][0]) == int(global_ids[6][0]) == 0


def test_identity_expires_outside_memory_window():
    global_ids = track([[A]] + [[]] * 8 + [[A]])
    assert int(global_ids[0][0]) == 0
    assert int(global_ids[9][0]) == 1


def test_two_objects_remain_distinct():
    global_ids = track([[A, B]] * 5)
    assert all(ids.tolist() == global_ids[0].tolist() for ids in global_ids)


def test_vocabulary_exhaustion_does_not_recycle_live_identity():
    global_ids = track([[A, B], [A, B], [D, A, B]], vocabulary_size=2)
    assert int(global_ids[2][0]) == -1


def test_dead_relative_slot_reuse_mints_new_global_identity():
    global_ids = track([[A]] + [[]] * 8 + [[B], [B]], vocabulary_size=1)
    assert (int(global_ids[0][0]), int(global_ids[9][0]), int(global_ids[10][0])) == (0, 1, 1)
