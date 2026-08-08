import pytest

from sparseqid.cli.main import main as cli_main
from sparseqid.cli.train import PAPER_SCENES, parser
from sparseqid.constants import MODEL_IDX_TO_CLASS_ID


def test_model_class_mapping_is_submission_mapping() -> None:
    assert MODEL_IDX_TO_CLASS_ID == {0: 0, 1: 4, 2: 5, 3: 2, 4: 3, 5: 1, 6: 6}


def test_training_defaults_are_the_submitted_frozen_recipe() -> None:
    args = parser().parse_args(["--base-checkpoint", "base.pth"])
    assert len(PAPER_SCENES) == 35
    assert args.iterations == 6000
    assert args.clip_length == 30
    assert (args.interval_min, args.interval_max) == (1, 4)
    assert args.augmentation_groups == 6
    assert args.vocabulary_size == 128
    assert args.position_encoding == "fourier"
    assert args.learning_rate == 4e-4
    assert args.gradient_clip == 1.0


def test_unified_cli_lists_the_public_commands(capsys) -> None:
    cli_main([])
    help_text = capsys.readouterr().out
    assert all(
        command in help_text for command in ("extract", "train", "infer", "visualize")
    )
    assert "evaluate" not in help_text


def test_unified_cli_delegates_subcommand_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["train", "--help"])
    assert exit_info.value.code == 0
    assert "usage: sqid train" in capsys.readouterr().out
