"""Check named group inheritance, strict removal of old syntax and independent epochs."""

from collections import Counter

import pytest

from unituner.config.loading import ConfigError, load_training_config
from unituner.data.datasets.batching import DatasetBatches, EpochBatches


def document():
    return {
        "model": {"family": "z-image", "variant": "base", "weights": "model.safetensors"},
        "train": {"adapter": {}},
        "dataset": {
            "defaults": {"batch_size": 2, "resolution": [512, 512], "repeats": 2, "caption_extension": ".caption"},
            "groups": {
                "portraits": {"source": "portraits", "cache_directory": "cache/portraits"},
                "landscapes": {
                    "source": "landscapes",
                    "cache_directory": "cache/landscapes",
                    "batch_size": 1,
                    "resolution": [768, 512],
                    "repeats": 1,
                    "enable_bucket": False,
                },
            },
        },
    }


def test_groups_inherit_defaults_and_explicit_values_win():
    config = load_training_config(overrides=document())
    portraits, landscapes = config.dataset.groups.values()
    assert (portraits.batch_size, portraits.resolution, portraits.repeats) == (2, (512, 512), 2)
    assert (landscapes.batch_size, landscapes.resolution, landscapes.repeats) == (1, (768, 512), 1)
    assert portraits.caption_extension == landscapes.caption_extension == ".caption"
    assert not landscapes.enable_bucket
    assert type(config).model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": 0},
        {"resolution": [0, 512]},
        {"repeats": 0},
        {"batch_siz": 2},
    ],
)
def test_each_group_is_validated(changes):
    values = document()
    values["dataset"]["groups"]["landscapes"].update(changes)
    with pytest.raises(ConfigError, match="landscapes"):
        load_training_config(overrides=values)


@pytest.mark.parametrize("name", ["", "with-dash", "with.dot", "123", "风景"])
def test_group_names_remain_addressable_in_cli(name):
    values = document()
    values["dataset"]["groups"][name] = values["dataset"]["groups"].pop("landscapes")
    with pytest.raises(ConfigError, match="group names"):
        load_training_config(overrides=values)


def test_cache_aliases_are_rejected(tmp_path):
    values = document()
    values["dataset"]["groups"]["portraits"]["cache_directory"] = str(tmp_path / "cache")
    values["dataset"]["groups"]["landscapes"]["cache_directory"] = str(tmp_path / "nested" / ".." / "cache")
    with pytest.raises(ConfigError, match="distinct resolved cache_directory"):
        load_training_config(overrides=values)


@pytest.mark.parametrize(
    "old",
    [
        {"source": "images", "cache_directory": "cache"},
        {"sources": [{"source": "images", "cache_directory": "cache"}]},
    ],
)
def test_obsolete_dataset_configuration_is_rejected(old):
    values = document()
    values["dataset"] = old
    with pytest.raises(ConfigError, match="dataset"):
        load_training_config(overrides=values)


def test_video_groups_validate_each_groups_model_constraints():
    values = document()
    values["model"] = {"family": "wan", "variant": "t2v", "version": "2.1", "weights": "wan.safetensors"}
    values["dataset"]["groups"]["landscapes"].update({"kind": "video", "video": {"target_frames": [17, 33]}})
    video = load_training_config(overrides=values).dataset.groups["landscapes"].video
    assert video is not None
    assert video.target_frames == (17, 33)
    values["dataset"]["groups"]["landscapes"]["video"]["target_frames"] = [16]
    with pytest.raises(ConfigError, match=r"dataset.groups.landscapes.*1\+4k"):
        load_training_config(overrides=values)


def test_independent_batches_preserve_repeats_short_tails_and_resume():
    def plan():
        return DatasetBatches(
            {
                "portraits": EpochBatches({(0,): ["p0", "p1", "p2"] * 2}, batch_size=4, seed=42),
                "landscapes": EpochBatches({(0,): ["l0", "l1", "l2"]}, batch_size=2, seed=42),
            },
            seed=42,
        )

    continuous = plan()
    assert (continuous.num_batches, continuous.num_samples) == (4, 9)
    first = [continuous.batch(index) for index in range(4)]
    assert sorted(len(batch) for name, batch in first if name == "portraits") == [2, 4]
    assert sorted(len(batch) for name, batch in first if name == "landscapes") == [1, 2]
    assert Counter(value for _, batch in first for value in batch) == Counter(
        {"p0": 2, "p1": 2, "p2": 2, "l0": 1, "l1": 1, "l2": 1}
    )
    assert all(value.startswith(name[0]) for name, batch in first for value in batch)
    expected = [continuous.batch(index) for index in range(20)]
    resumed = plan()
    assert [resumed.batch(index) for index in range(7, 20)] == expected[7:]
    assert resumed.samples_before(13) == sum(len(batch) for _, batch in expected[:13])
    assert expected[:4] != expected[4:8]


def test_group_order_is_part_of_resume_identity():
    from unituner.checkpoints.schema import build_resume_identity

    values = document()
    first = load_training_config(overrides=values)
    values["dataset"]["groups"] = dict(reversed(list(values["dataset"]["groups"].items())))
    second = load_training_config(overrides=values)
    assert build_resume_identity(
        first, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64
    ) != build_resume_identity(second, dataset_manifest_sha256="a" * 64, model_artifact_sha256="b" * 64)
