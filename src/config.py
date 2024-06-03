import yaml
from pathlib import Path


class Config:
    def __init__(self, config_file):
        self._config_file = config_file
        self._config = self._load_config()

    def _load_config(self):
        with open(self._config_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def __getattr__(self, item):
        if item in self._config:
            return self._config[item]
        else:
            raise AttributeError(f"'Config' object has no attribute '{item}'")


config = Config(Path(__file__).parent / 'config.yml')
