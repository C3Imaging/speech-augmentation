import yaml
from schema import Schema, Or, And, Optional, SchemaError


# schema defining the structure that 'asr/multi_asr_dataset_creator_funnel_config.yaml' must follow.
schema_multi_asr_config = Schema(
    {
        "output_folder": str,
        "speech_segments_funnels_thresholds": And(
            [float], lambda vals: len(vals) > 1
        ),  # must be a list of floats with at least 2 elements.
        "decoder_folders": [str],  # must be a list of strings.
        "hypothesis_level_weights": [
            lambda n: 0.0 <= n <= 1.0
        ],  # must be a list of floats in range 0.0<->1.0
        "algorithm_type": Or(
            "token_based_depthwise_algo", "hypothesis_based_lengthwise_algo"
        ),
        "compound_score_threshold": float,
        "iou_tokens_time_threshold": And(float, lambda n: 0.0 < n < 1.0),
        "speech_segment_length": And(float, lambda n: n > 0.0),
    }
)


class Config(object):
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of nested elements, e.g. cfg.get_config('meta/dataset_name')"""

    def __init__(self, config_path):
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())

    def validate_data(self, config_schema):
        try:
            config_schema.validate(self._data)
            print("Config data is valid.")
        except SchemaError as se:
            raise se

    def get(self, path=None, default=None):
        """Parses the string, e.g. 'experiment/training/batch_size' by splitting it into a list and recursively accessing the nested sub-dictionaries."""
        # we need to deep-copy self._data to avoid over-writing its data
        sub_dict = dict(self._data)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default

    def add(self, key, value):
        """Add an object to self._data"""
        self._data[key] = value


cfg = None


def init(cfg_path, schema):
    global cfg
    # create config object from yaml file.
    cfg = Config(cfg_path)
    # validate the fields of the yaml file loaded as the data in the Config object against the applicable schema.
    cfg.validate_data(schema)
