from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from nl2sql.config import RunConfig, load_config

CONFIGS = sorted(Path("configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_shipped_configs_load(path: Path) -> None:
    config = load_config(path)

    assert config.name == path.stem and config.thinking is False


def base() -> dict:
    return yaml.safe_load(Path("configs/base_9b_local.yaml").read_text()) | {"name": "t"}


@pytest.mark.parametrize(
    "change",
    [
        {"thinking": True},
        {"guard": {"inject_limit": True}},
        {"linking": {"enabled": True}},  # not implemented yet: must not be silently ignored
        {"prompt": {"schema_token_budget": 3072, "max_total_tokens": 9000}},  # over the context window
        {"unknown_key": 1},
    ],
)
def test_invalid_configs_are_rejected(change: dict) -> None:
    with pytest.raises(ValidationError):
        RunConfig(**(base() | change))
