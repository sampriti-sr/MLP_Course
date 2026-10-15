"""Loaders for the real-world datasets used throughout the MLP course.

>>> from mlp_data import load_retail
>>> df = load_retail()
"""
from . import models, retail, serving, style, tracking
from ._download import data_dir
from .loaders import (
    DATASETS,
    MovieLens,
    load_airbnb,
    load_credit,
    load_movielens,
    load_retail,
    load_taxi,
)

__all__ = [
    "DATASETS", "MovieLens", "data_dir", "models", "retail", "serving", "style", "tracking",
    "load_airbnb", "load_credit", "load_movielens", "load_retail", "load_taxi",
]
__version__ = "0.1.0"
