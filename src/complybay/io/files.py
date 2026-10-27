"""YAML configuration files and column data files.

Data files are whitespace-separated text with ``#`` comments, one sample per
row and three columns ``x  value  sigma``:

| data | x | value | sigma |
|---|---|---|---|
| compliance | frequency (Hz) | normalized compliance (1/Pa) | 1/Pa |
| phase velocity | period (s) | km/s | km/s |
| ACF | lag (s), from 0 every dt | ACF | ACF units |
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level.")
    return data


def dump_yaml(data: dict, path: str | Path) -> None:
    with open(path, "w") as stream:
        yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True)


@dataclass(frozen=True)
class ColumnData:
    x: np.ndarray
    value: np.ndarray
    sigma: np.ndarray
    path: Path


def read_three_columns(path: str | Path) -> ColumnData:
    """Read ``x value sigma`` rows; x must increase, sigma must be positive."""
    path = Path(path)
    try:
        table = np.loadtxt(path, comments="#", ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{path}: not a numeric three-column file ({exc}).") from None
    if table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError(f"{path}: expected three columns (x value sigma), got {table.shape[1]}.")
    x, value, sigma = table.T
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"{path}: the first column must increase strictly.")
    if np.any(sigma <= 0):
        raise ValueError(f"{path}: sigma (third column) must be positive.")
    return ColumnData(x=x, value=value, sigma=sigma, path=path)


def write_three_columns(path: str | Path, x, value, sigma, *, header: str) -> None:
    """Write a data file in the format :func:`read_three_columns` reads."""
    np.savetxt(path, np.column_stack((x, value, sigma)), fmt="%.10g", header=header)
