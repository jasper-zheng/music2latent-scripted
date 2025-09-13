import importlib.util
from hparams import hparams, HParams

def load_config(config_path: str) -> HParams:
    """Loads a configuration from a Python file.

    Args:
        config_path: Path to the Python configuration file.

    Returns:
        An updated HParams object
    """
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    config_dict = {
        k: v for k, v in config_module.__dict__.items() if not k.startswith("__")
    }
    hparams.update(config_dict)
    return hparams