"""Default no-op batch transform used as the data.batch target in configs."""

from typing import Any, Dict


class IdentityBatchTransform:
    """Pass-through batch transform: returns the batch dictionary unchanged."""

    def __init__(self, **kwargs: Any):
        self.params = kwargs

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"
