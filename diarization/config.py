import yaml

class Config(object):  
    """Simple dict wrapper that adds a thin API allowing for slash-based retrieval of nested elements, e.g. cfg.get_config("meta/dataset_name")"""
    
    def __init__(self, config_path):
        with open(config_path) as cf_file:
            self._data = yaml.safe_load(cf_file.read())
    
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