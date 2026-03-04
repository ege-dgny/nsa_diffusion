"""Per-method default configurations."""

from __future__ import annotations

from configs import ExperimentConfig


DEFAULTS: dict[str, dict] = {
    "lowrank_kd": {
        "alpha": 0.0,
        "alpha_s": 0.0,
        "beta": 0.1,
        "lam": 0.1,
    },
    "standard_nsa": {
        "alpha": 0.1,
        "alpha_s": 0.0,
        "beta": 0.1,
        "lam": 0.1,
    },
    "nsa_diff": {
        "alpha": 0.1,
        "alpha_s": 0.1,
        "beta": 0.1,
        "lam": 0.1,
    },
    "fitnets": {
        "alpha": 0.0,
        "alpha_s": 0.0,
        "beta": 0.1,
        "lam": 0.0,
    },
    "gramian": {
        "alpha": 0.0,
        "alpha_s": 0.0,
        "beta": 0.1,
        "lam": 0.0,
    },
}


def get_default_config(method: str, **overrides) -> ExperimentConfig:
    """Get default config for a method with optional overrides."""
    base = DEFAULTS.get(method, {})
    base["method"] = method
    base.update(overrides)
    return ExperimentConfig(**base)
