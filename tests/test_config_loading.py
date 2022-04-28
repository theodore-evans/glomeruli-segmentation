import pytest
from marshmallow import ValidationError

from glomeruli_segmentation.config import Config, load_config
from glomeruli_segmentation.data_classes import BlendMode


def test_that_a_default_config_is_created_when_config_file_is_blank():
    default_config = Config()
    loaded_config = load_config("")
    assert default_config == loaded_config


def test_that_an_error_is_raised_when_config_file_does_not_exist():
    invalid_filepath = "invalid_filepath"
    with pytest.raises(FileNotFoundError):
        load_config(invalid_filepath)


def test_that_a_config_is_created_when_config_file_exists():
    config_filepath = "tests/config_valid.json"
    loaded_config = load_config(config_filepath)
    assert isinstance(loaded_config, Config)
    assert loaded_config.blend_mode == BlendMode.MEAN


def test_that_a_config_with_invalid_values_raises_a_validation_error():
    config_filepath = "tests/config_invalid.json"
    with pytest.raises(ValidationError):
        load_config(config_filepath)
