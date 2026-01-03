import yaml
import os

class Config:
    _instance = None
    _config_data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self, path="config.yaml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found at: {path}")
        
        with open(path, 'r') as f:
            self._config_data = yaml.safe_load(f)

    @property
    def data(self):
        return self._config_data

    def get(self, *keys):
        """Recursive get for nested keys"""
        data = self._config_data
        for key in keys:
            data = data.get(key)
            if data is None:
                return None
        return data

# Singleton instance
cfg = Config().data