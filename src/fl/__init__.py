"""FL module."""

def __getattr__(name):
    """Lazy imports to avoid requiring flwr when only using manual loop."""
    if name == "FederatedClient":
        from src.fl.client import FederatedClient
        return FederatedClient
    if name in ("WeightedFedAvg", "FedProxStrategy", "TrimmedMeanStrategy", "build_strategy"):
        from src.fl.strategy import WeightedFedAvg, FedProxStrategy, TrimmedMeanStrategy, build_strategy
        return globals()[name]
    if name == "run_manual_simulation":
        from src.fl.manual_loop import run_manual_simulation
        return run_manual_simulation
    if name == "protect_update":
        from src.fl.secure_update import protect_update
        return protect_update
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FederatedClient",
    "WeightedFedAvg", "FedProxStrategy", "TrimmedMeanStrategy", "build_strategy",
    "run_manual_simulation",
    "protect_update",
]
