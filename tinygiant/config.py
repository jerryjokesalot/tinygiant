import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TinyGiantConfig:
    model_path: str = ""
    cache_dir: str = ""
    pin_count: int = 48
    calibrate_tokens: int = 10
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self):
        if self.model_path:
            self.model_path = os.path.expanduser(self.model_path)
        if self.cache_dir:
            self.cache_dir = os.path.expanduser(self.cache_dir)

    @classmethod
    def from_file(cls, path):
        import json
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_file(self, path):
        import json
        from dataclasses import asdict
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
