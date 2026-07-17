from typing import Dict, Any, Callable

class ExperimentRegistry:
    """
    Registers experiments with corresponding configurations and executable entrypoints.
    """
    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, config: Dict[str, Any], entrypoint: Callable) -> None:
        """
        Registers a new experiment configuration and execution logic.
        """
        if name in self._registry:
            raise ValueError(f"Experiment {name} is already registered.")
        self._registry[name] = {
            "config": config,
            "entrypoint": entrypoint
        }

    def get(self, name: str) -> Dict[str, Any]:
        """
        Retrieves a registered experiment mapping.
        """
        if name not in self._registry:
            raise KeyError(f"Experiment {name} is not registered.")
        return self._registry[name]

    def list_experiments(self) -> list:
        """
        Returns a list of all registered experiment names.
        """
        return list(self._registry.keys())
