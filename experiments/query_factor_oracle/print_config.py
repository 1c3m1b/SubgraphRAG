from __future__ import annotations

import argparse
from typing import Any

from .qforacle.config import load_config


def _lookup(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a resolved experiment config field")
    parser.add_argument("--config", required=True)
    parser.add_argument("--field", required=True)
    args = parser.parse_args()
    print(_lookup(load_config(args.config), args.field))


if __name__ == "__main__":
    main()
