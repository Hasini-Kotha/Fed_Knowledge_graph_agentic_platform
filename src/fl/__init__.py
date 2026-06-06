"""FL module — lazy imports to avoid loading Flower unless needed."""


def __getattr__(name):
    if name == "FederatedClient":
        from src.fl.client import FederatedClient
        return FederatedClient
    if name == "FedProxStrategy":
        from src.fl.strategy import FedProxStrategy
        return FedProxStrategy
    if name in ("create_server_config", "build_fedprox_strategy", "run_standalone_server"):
        from src.fl.server import create_server_config, build_fedprox_strategy, run_standalone_server
        if name == "create_server_config":
            return create_server_config
        elif name == "build_fedprox_strategy":
            return build_fedprox_strategy
        else:
            return run_standalone_server
    if name == "protect_update":
        from src.fl.secure_update import protect_update
        return protect_update
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FederatedClient",
    "FedProxStrategy",
    "create_server_config",
    "build_fedprox_strategy",
    "run_standalone_server",
    "protect_update",
]
