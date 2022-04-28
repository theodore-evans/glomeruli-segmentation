import json
from dataclasses import asdict, dataclass
from logging import Logger
from typing import Tuple

import desert

from glomeruli_segmentation.data_classes import BlendMode
from glomeruli_segmentation.logging_tools import get_logger


@dataclass
class Config:
    slide_key: str = "slide"
    roi_key: str = "region_of_interest"
    results_key_stub: str = "glomeruli_segmentation"
    ead_namespace: str = "org.empaia.dai.glomeruli_segmentation.v1"
    binary_threshold: float = 0.70
    normal_class_suffix: str = "glomerulus"
    anomaly_class_suffix: str = "anomaly"
    anomaly_confidence_threshold: float = 0.70

    torch_vision_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    torch_vision_std: Tuple[float, float, float] = (0.485, 0.456, 0.406)

    window_size: Tuple[int, int] = (1024, 1024)
    stride: Tuple[int, int] = (1024, 1024)
    blend_mode: BlendMode = BlendMode.MEAN

    def __str__(self):
        return json.dumps(asdict(self), indent=4, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "Config":
        return cls(**json.loads(json_str))

    def _get_class_namespace(self, class_suffix: str) -> str:
        return f"{self.ead_namespace}.classes.{class_suffix}"

    @property
    def normal_class(self) -> str:
        return self._get_class_namespace(self.normal_class_suffix)

    @property
    def anomaly_class(self) -> str:
        return self._get_class_namespace(self.anomaly_class_suffix)


# TODO: load configuration from {JOB_ID}/configuration endpoint, if available"
# Blocked by limited configuration data types, i.e. no type for normalization mean, std arrays
def load_config(config_file: str, logger: Logger = get_logger()) -> Config:
    """
    Loads the configuration file.

    :param config_file: Path to the configuration file.
    :param logger: Logger to use for logging.
    """
    if not config_file:
        logger.info("No configuration file provided, using default")
        config = Config()
    else:
        try:
            schema = desert.schema(Config)
            with open(config_file, "r", encoding="utf-8") as f:
                config = schema.load(json.load(f))
            logger.info(f"Using config file: {config_file}")
        except Exception as e:
            logger.error(f"Error while loading config file '{config_file}': {e}")
            raise e
    logger.debug(f"config={config}")
    return config
